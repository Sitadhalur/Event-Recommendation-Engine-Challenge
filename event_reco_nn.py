#!/usr/bin/env python3
"""PyTorch neural cold-start ranker for the event recommendation project."""

from __future__ import annotations

import argparse
import os
import random

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from event_reco_baseline import DATA_DIR, build_everything, map_at_k, parse_int, write_submission
from event_reco_sklearn import score_rows, selected_feature_indices


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_by_user(train_rows):
    users = sorted({row["user"] for row in train_rows}, key=lambda x: int(x))
    valid_users = set(users[int(len(users) * 0.8) :])
    train_fit = [row for row in train_rows if row["user"] not in valid_users]
    valid = [row for row in train_rows if row["user"] in valid_users]
    return train_fit, valid


def build_category_maps(rows):
    locales = {"": 0}
    genders = {"": 0}
    countries = {"": 0}
    months = {"": 0}
    for row in rows:
        user_info = row.get("_user_info", {})
        event_info = row.get("_event_info", {})
        locale = user_info.get("locale", "")
        gender = user_info.get("gender", "")
        country = event_info.get("country", "")
        start = event_info.get("start")
        month = str(start.month) if start else ""
        for mapping, value in [(locales, locale), (genders, gender), (countries, country), (months, month)]:
            if value not in mapping:
                mapping[value] = len(mapping)
    return locales, genders, countries, months


def attach_info(rows, builder):
    out = []
    for row in rows:
        new = dict(row)
        new["_user_info"] = builder.users.get(row["user"], {})
        new["_event_info"] = builder.events.get(row["event"], {})
        out.append(new)
    return out


def vectorize(builder, rows, indices, cat_maps=None, fit_cats=False):
    rows = attach_info(rows, builder)
    x = np.asarray([builder.transform_row(row) for row in rows], dtype=np.float32)[:, indices]
    words = np.asarray([builder.events.get(row["event"], {}).get("words", [0] * 100) for row in rows], dtype=np.float32)
    words = np.log1p(words)
    y = np.asarray([parse_int(row.get("interested", "0"), 0) for row in rows], dtype=np.float32)
    if cat_maps is None:
        cat_maps = build_category_maps(rows)
    locales, genders, countries, months = cat_maps
    cat = np.asarray(
        [
            [
                locales.get(row["_user_info"].get("locale", ""), 0),
                genders.get(row["_user_info"].get("gender", ""), 0),
                countries.get(row["_event_info"].get("country", ""), 0),
                months.get(str(row["_event_info"].get("start").month) if row["_event_info"].get("start") else "", 0),
            ]
            for row in rows
        ],
        dtype=np.int64,
    )
    return x, words, cat, y, cat_maps


class DeepColdRanker(nn.Module):
    def __init__(self, n_num, cat_sizes, hidden=128, dropout=0.2):
        super().__init__()
        emb_dims = [min(16, max(2, int(size**0.25 * 4))) for size in cat_sizes]
        self.embs = nn.ModuleList([nn.Embedding(size, dim) for size, dim in zip(cat_sizes, emb_dims)])
        self.word_net = nn.Sequential(
            nn.Linear(100, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        total = n_num + sum(emb_dims) + 32
        self.net = nn.Sequential(
            nn.Linear(total, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x_num, x_words, x_cat):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        word_vec = self.word_net(x_words)
        x = torch.cat([x_num, word_vec] + embs, dim=1)
        return self.net(x).squeeze(1)


def make_loader(x, words, cat, y, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(words, dtype=torch.float32),
        torch.tensor(cat, dtype=torch.long),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def predict(model, x, words, cat, batch_size):
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, wb, cb, _ in make_loader(x, words, cat, np.zeros(len(x), dtype=np.float32), batch_size, False):
            preds.append(torch.sigmoid(model(xb, wb, cb)).cpu().numpy())
    return np.concatenate(preds)


def train_model(x_train, w_train, c_train, y_train, x_valid, w_valid, c_valid, y_valid, valid_rows, cat_sizes, args):
    model = DeepColdRanker(x_train.shape[1], cat_sizes, hidden=args.hidden, dropout=args.dropout)
    pos = float(y_train.sum())
    neg = float(len(y_train) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_score = -1.0
    best_state = None
    patience = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, wb, cb, yb in make_loader(x_train, w_train, c_train, y_train, args.batch_size, True):
            opt.zero_grad()
            loss = loss_fn(model(xb, wb, cb), yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        pred = predict(model, x_valid, w_valid, c_valid, args.batch_size)
        score = map_at_k(score_rows(valid_rows, y_valid.astype(int), pred))
        print(f"epoch={epoch:02d} loss={np.mean(losses):.5f} valid_MAP={score:.6f}")
        if score > best_score:
            best_score = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
        if patience >= args.patience:
            break
    model.load_state_dict(best_state)
    return model, best_score


def run(args):
    set_seed(args.seed)
    torch.set_num_threads(args.threads)
    train_rows, test_rows, builder = build_everything()
    indices = selected_feature_indices(builder.feature_names, "cold")
    fit_rows, valid_rows = split_by_user(train_rows)
    x_fit, w_fit, c_fit, y_fit, cat_maps = vectorize(builder, fit_rows, indices)
    x_valid, w_valid, c_valid, y_valid, _ = vectorize(builder, valid_rows, indices, cat_maps)

    scaler = StandardScaler()
    x_fit = scaler.fit_transform(x_fit).astype(np.float32)
    x_valid = scaler.transform(x_valid).astype(np.float32)
    cat_sizes = [len(m) for m in cat_maps]
    print(f"NN features numeric={x_fit.shape[1]} cat_sizes={cat_sizes} train={len(x_fit)} valid={len(x_valid)}")
    model, best = train_model(x_fit, w_fit, c_fit, y_fit, x_valid, w_valid, c_valid, y_valid, valid_rows, cat_sizes, args)
    print(f"Best validation MAP@200: {best:.6f}")

    print("Retraining on full train...")
    x_all, w_all, c_all, y_all, cat_maps = vectorize(builder, train_rows, indices)
    x_test, w_test, c_test, _, _ = vectorize(builder, test_rows, indices, cat_maps)
    full_scaler = StandardScaler()
    x_all = full_scaler.fit_transform(x_all).astype(np.float32)
    x_test = full_scaler.transform(x_test).astype(np.float32)
    full_model, _ = train_model(
        x_all,
        w_all,
        c_all,
        y_all,
        x_all,
        w_all,
        c_all,
        y_all,
        train_rows,
        [len(m) for m in cat_maps],
        args,
    )
    pred_test = predict(full_model, x_test, w_test, c_test, args.batch_size)
    scored = [(row["user"], row["event"], float(score)) for row, score in zip(test_rows, pred_test)]
    write_submission(DATA_DIR / args.output, scored)
    write_submission(DATA_DIR / args.output.replace(".csv", "_legacy.csv"), scored, legacy=True)
    print(f"Wrote {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission_nn.csv")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
