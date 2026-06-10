#!/usr/bin/env python3
"""
Zero-dependency baseline for Kaggle Event Recommendation Engine Challenge.

The goal of this script is to prove the local data pipeline is healthy:
read all provided files, build useful first-pass features, train a small
logistic model, validate with MAP@200, and write a submission file.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import math
import random
import re
import unicodedata
import zipfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    from rapidfuzz import fuzz, process
except Exception:
    fuzz = None
    process = None


DATA_DIR = Path(__file__).resolve().parent
K = 200
GEO_CACHE = None


def normalize_geo_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", value or "")
    value = value.encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", " ", value).lower()
    return re.sub(r"\s+", " ", value).strip()


def haversine_km(lat1, lng1, lat2, lng2) -> float:
    if not lat1 or not lng1 or not lat2 or not lng2:
        return 0.0
    r = 6371.0
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_geonames(data_dir: Path):
    global GEO_CACHE
    if GEO_CACHE is not None:
        return GEO_CACHE
    country_path = data_dir / "countryInfo.txt"
    cities_path = data_dir / "cities5000.zip"
    if not country_path.exists() or not cities_path.exists():
        return None

    country_alias = {}
    with country_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 17:
                continue
            iso, iso3, _, _, country_name = parts[:5]
            aliases = {country_name, iso, iso3}
            aliases.update(country_name.replace(",", " ").split())
            for alias in aliases:
                key = normalize_geo_text(alias)
                if len(key) >= 2:
                    country_alias[key] = iso

    cities_by_country = defaultdict(list)
    city_choices_by_country = defaultdict(list)
    city_choices_global = []
    with zipfile.ZipFile(cities_path) as zf:
        with zf.open("cities5000.txt") as raw:
            for bline in raw:
                line = bline.decode("utf-8", "replace").rstrip("\n")
                parts = line.split("\t")
                if len(parts) < 15:
                    continue
                name = parts[1]
                asciiname = parts[2]
                alternates = parts[3].split(",")[:12]
                lat = parse_float(parts[4], 0.0)
                lng = parse_float(parts[5], 0.0)
                country = parts[8]
                population = parse_int(parts[14], 0)
                aliases = {name, asciiname}
                aliases.update(a for a in alternates if 3 <= len(a) <= 40)
                record = {
                    "name": normalize_geo_text(asciiname or name),
                    "country": country,
                    "lat": lat,
                    "lng": lng,
                    "population": population,
                    "aliases": [normalize_geo_text(a) for a in aliases if normalize_geo_text(a)],
                }
                cities_by_country[country].append(record)
    for country, records in cities_by_country.items():
        records.sort(key=lambda r: r["population"], reverse=True)
        # Keep matching lists compact; small towns add noise for this dataset.
        for record in records[:2500]:
            for alias in record["aliases"]:
                if len(alias) >= 3:
                    city_choices_by_country[country].append((alias, record))
                    city_choices_global.append((alias, record))
    GEO_CACHE = {
        "country_alias": country_alias,
        "cities_by_country": cities_by_country,
        "city_choices_by_country": city_choices_by_country,
        "city_choices_global": city_choices_global,
    }
    return GEO_CACHE


def match_country(text: str, geo) -> str:
    if not text or not geo:
        return ""
    padded = f" {normalize_geo_text(text)} "
    best = ""
    best_len = 0
    for alias, code in geo["country_alias"].items():
        if len(alias) > best_len and f" {alias} " in padded:
            best = code
            best_len = len(alias)
    return best


def match_city(text: str, country_code: str, geo):
    if not text or not geo:
        return None
    norm = normalize_geo_text(text)
    padded = f" {norm} "
    choices = geo["city_choices_by_country"].get(country_code) if country_code else None
    if not choices:
        choices = geo["city_choices_global"]
    best_record = None
    best_pop = -1
    for alias, record in choices:
        if len(alias) >= 3 and f" {alias} " in padded and record["population"] > best_pop:
            best_record = record
            best_pop = record["population"]
    if best_record:
        return best_record
    if process is None or fuzz is None or len(norm) < 4:
        return None
    names = [alias for alias, _ in choices]
    found = process.extractOne(norm, names, scorer=fuzz.token_set_ratio, score_cutoff=88 if country_code else 93)
    if not found:
        return None
    return choices[found[2]][1]


def parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    value = value.replace("T", " ")
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def parse_int(value: str, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def split_ids(value: str) -> list[str]:
    value = (value or "").strip()
    if not value:
        return []
    return value.split()


def read_interactions(path: Path, has_labels: bool) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if has_labels:
        for row in rows:
            row["label"] = row["interested"]
    return rows


def collect_candidate_sets(train_rows: list[dict[str, str]], test_rows: list[dict[str, str]]):
    candidate_events = set()
    candidate_users = set()
    for row in train_rows + test_rows:
        candidate_users.add(row["user"])
        candidate_events.add(row["event"])
    return candidate_users, candidate_events


def load_users(path: Path, candidate_users: set[str]) -> dict[str, dict[str, object]]:
    users = {}
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            user = row["user_id"]
            if user not in candidate_users:
                continue
            joined = parse_dt(row.get("joinedAt", ""))
            birthyear = parse_int(row.get("birthyear", ""), 0)
            location = (row.get("location") or "").lower()
            users[user] = {
                "locale": row.get("locale") or "",
                "birthyear": birthyear,
                "age_2012": 2012 - birthyear if 1900 <= birthyear <= 2012 else 0,
                "gender": row.get("gender") or "",
                "joined": joined,
                "location": location,
                "timezone": parse_int(row.get("timezone", ""), 0),
            }
    return users


def load_friends(path: Path, candidate_users: set[str]) -> tuple[dict[str, set[str]], dict[str, int]]:
    friends = {}
    friend_counts = {}
    with gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            user = row["user"]
            if user not in candidate_users:
                continue
            values = split_ids(row.get("friends", ""))
            friends[user] = set(values)
            friend_counts[user] = len(values)
    return friends, friend_counts


def load_event_attendees(
    path: Path,
    candidate_events: set[str],
    friends: dict[str, set[str]],
    candidate_pairs: set[tuple[str, str]],
) -> tuple[dict[str, dict[str, object]], dict[tuple[str, str], dict[str, int]]]:
    event_stats = {}
    pair_friend_stats = {}

    event_to_users = defaultdict(list)
    for user, event in candidate_pairs:
        event_to_users[event].append(user)

    with gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            event = row["event"]
            if event not in candidate_events:
                continue
            yes = set(split_ids(row.get("yes", "")))
            maybe = set(split_ids(row.get("maybe", "")))
            invited = set(split_ids(row.get("invited", "")))
            no = set(split_ids(row.get("no", "")))
            total = len(yes) + len(maybe) + len(invited) + len(no)
            event_stats[event] = {
                "yes": len(yes),
                "maybe": len(maybe),
                "att_invited": len(invited),
                "no": len(no),
                "total": total,
                "yes_ratio": len(yes) / total if total else 0.0,
                "no_ratio": len(no) / total if total else 0.0,
            }
            for user in event_to_users.get(event, []):
                fset = friends.get(user, set())
                if not fset:
                    continue
                pair_friend_stats[(user, event)] = {
                    "friend_yes": len(fset & yes),
                    "friend_maybe": len(fset & maybe),
                    "friend_invited": len(fset & invited),
                    "friend_no": len(fset & no),
                }
    return event_stats, pair_friend_stats


def load_events(path: Path, candidate_events: set[str]) -> dict[str, dict[str, object]]:
    events = {}
    with gzip.open(path, "rt", newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            event = row["event_id"]
            if event not in candidate_events:
                continue
            start = parse_dt(row.get("start_time", ""))
            words = []
            word_sum = 0
            word_max = 0
            for i in range(1, 101):
                value = parse_int((row.get(f"c_{i}") or "").strip(), 0)
                words.append(value)
                word_sum += value
                word_max = max(word_max, value)
            other = parse_int((row.get("c_other") or "").strip(), 0)
            word_sum += other
            city = (row.get("city") or "").lower()
            country = (row.get("country") or "").lower()
            events[event] = {
                "creator": row.get("user_id") or "",
                "start": start,
                "city": city,
                "country": country,
                "has_location": 1 if (row.get("lat") and row.get("lng")) else 0,
                "lat": parse_float(row.get("lat", ""), 0.0),
                "lng": parse_float(row.get("lng", ""), 0.0),
                "word_sum": word_sum,
                "word_max": word_max,
                "words": words,
            }
    return events


def build_user_history(train_rows: list[dict[str, str]], events: dict[str, dict[str, object]]):
    stats = defaultdict(lambda: Counter())
    event_label_stats = defaultdict(lambda: Counter())
    label_history_pairs = set()
    pos_word_sum = defaultdict(lambda: [0.0] * 100)
    pos_word_count = Counter()
    neg_word_sum = defaultdict(lambda: [0.0] * 100)
    neg_word_count = Counter()
    pos_events = defaultdict(set)
    neg_events = defaultdict(set)

    for row in train_rows:
        user = row["user"]
        interested = parse_int(row["interested"])
        not_interested = parse_int(row["not_interested"])
        invited = parse_int(row["invited"])
        stats[user]["rows"] += 1
        stats[user]["interested"] += interested
        stats[user]["not_interested"] += not_interested
        stats[user]["invited"] += invited
        event = row["event"]
        label_history_pairs.add((user, event))
        event_label_stats[event]["rows"] += 1
        event_label_stats[event]["interested"] += interested
        event_label_stats[event]["not_interested"] += not_interested
        event_label_stats[event]["invited"] += invited
        ev = events.get(row["event"])
        if not ev:
            continue
        words = ev["words"]
        if interested:
            pos_events[user].add(event)
            pos_word_count[user] += 1
            acc = pos_word_sum[user]
            for i, value in enumerate(words):
                acc[i] += value
        elif not_interested:
            neg_events[user].add(event)
            neg_word_count[user] += 1
            acc = neg_word_sum[user]
            for i, value in enumerate(words):
                acc[i] += value
    return (
        stats,
        pos_word_sum,
        pos_word_count,
        neg_word_sum,
        neg_word_count,
        pos_events,
        neg_events,
        event_label_stats,
        label_history_pairs,
    )


def build_friend_profiles(
    friends: dict[str, set[str]],
    pos_word_sum,
    pos_word_count,
    neg_word_sum,
    neg_word_count,
    pos_events=None,
    neg_events=None,
):
    friend_pos_sum = defaultdict(lambda: [0.0] * 100)
    friend_pos_count = Counter()
    friend_neg_sum = defaultdict(lambda: [0.0] * 100)
    friend_neg_count = Counter()
    friend_pos_event_scores = defaultdict(Counter)
    friend_neg_event_scores = defaultdict(Counter)
    for user, fset in friends.items():
        pos_acc = friend_pos_sum[user]
        neg_acc = friend_neg_sum[user]
        for friend in fset:
            pcount = pos_word_count.get(friend, 0)
            if pcount:
                friend_pos_count[user] += pcount
                profile = pos_word_sum[friend]
                for i, value in enumerate(profile):
                    pos_acc[i] += value
            ncount = neg_word_count.get(friend, 0)
            if ncount:
                friend_neg_count[user] += ncount
                profile = neg_word_sum[friend]
                for i, value in enumerate(profile):
                    neg_acc[i] += value
            if pos_events is not None:
                for event in pos_events.get(friend, ()):
                    friend_pos_event_scores[user][event] += 1.0
            if neg_events is not None:
                for event in neg_events.get(friend, ()):
                    friend_neg_event_scores[user][event] += 1.0
            for friend2 in friends.get(friend, ()) if pos_events is not None else ():
                # Two-hop signal is downweighted to keep direct friends dominant.
                for event in pos_events.get(friend2, ()):
                    friend_pos_event_scores[user][event] += 0.25
                if neg_events is not None:
                    for event in neg_events.get(friend2, ()):
                        friend_neg_event_scores[user][event] += 0.25
    return (
        friend_pos_sum,
        friend_pos_count,
        friend_neg_sum,
        friend_neg_count,
        friend_pos_event_scores,
        friend_neg_event_scores,
    )


def cosine_to_profile(words: list[int], profile: list[float], count: int) -> float:
    if count <= 0:
        return 0.0
    dot = 0.0
    a2 = 0.0
    b2 = 0.0
    for i, value in enumerate(words):
        b = profile[i] / count
        dot += value * b
        a2 += value * value
        b2 += b * b
    if a2 <= 0.0 or b2 <= 0.0:
        return 0.0
    return dot / math.sqrt(a2 * b2)


class FeatureBuilder:
    def __init__(
        self,
        users,
        friends,
        friend_counts,
        events,
        event_stats,
        pair_friend_stats,
        user_stats,
        pos_word_sum,
        pos_word_count,
        neg_word_sum,
        neg_word_count,
        pos_events=None,
        neg_events=None,
        event_label_stats=None,
        label_history_pairs=None,
        friend_pos_word_sum=None,
        friend_pos_word_count=None,
        friend_neg_word_sum=None,
        friend_neg_word_count=None,
        friend_pos_event_scores=None,
        friend_neg_event_scores=None,
    ):
        self.users = users
        self.friends = friends
        self.friend_counts = friend_counts
        self.events = events
        self.event_stats = event_stats
        self.pair_friend_stats = pair_friend_stats
        self.user_stats = user_stats
        self.pos_word_sum = pos_word_sum
        self.pos_word_count = pos_word_count
        self.neg_word_sum = neg_word_sum
        self.neg_word_count = neg_word_count
        self.pos_events = pos_events if pos_events is not None else {}
        self.neg_events = neg_events if neg_events is not None else {}
        self.event_label_stats = event_label_stats if event_label_stats is not None else defaultdict(Counter)
        self.label_history_pairs = label_history_pairs if label_history_pairs is not None else set()
        if friend_pos_word_sum is None:
            friend_profiles = build_friend_profiles(
                friends,
                pos_word_sum,
                pos_word_count,
                neg_word_sum,
                neg_word_count,
                self.pos_events,
                self.neg_events,
            )
            (
                friend_pos_word_sum,
                friend_pos_word_count,
                friend_neg_word_sum,
                friend_neg_word_count,
                friend_pos_event_scores,
                friend_neg_event_scores,
            ) = friend_profiles
        self.friend_pos_word_sum = friend_pos_word_sum
        self.friend_pos_word_count = friend_pos_word_count
        self.friend_neg_word_sum = friend_neg_word_sum
        self.friend_neg_word_count = friend_neg_word_count
        self.friend_pos_event_scores = friend_pos_event_scores if friend_pos_event_scores is not None else defaultdict(Counter)
        self.friend_neg_event_scores = friend_neg_event_scores if friend_neg_event_scores is not None else defaultdict(Counter)
        self.geo = load_geonames(DATA_DIR)
        self.user_geo = {}
        if self.geo:
            for user, info in users.items():
                location = info.get("location", "")
                country_code = match_country(location, self.geo)
                city = match_city(location, country_code, self.geo)
                self.user_geo[user] = {
                    "country": country_code or (city["country"] if city else ""),
                    "city": city["name"] if city else "",
                    "lat": city["lat"] if city else 0.0,
                    "lng": city["lng"] if city else 0.0,
                }
        self.creator_counts = Counter(ev.get("creator", "") for ev in events.values() if ev.get("creator", ""))
        self.event_city_counts = Counter(ev.get("city", "") for ev in events.values() if ev.get("city", ""))
        self.event_country_counts = Counter(ev.get("country", "") for ev in events.values() if ev.get("country", ""))
        self.feature_names = [
            "invited",
            "time_to_event_days",
            "log_positive_time_to_event",
            "event_within_1_day",
            "event_within_7_days",
            "event_within_30_days",
            "event_weekend",
            "event_month",
            "notify_hour",
            "event_has_location",
            "same_city_hint",
            "same_country_hint",
            "geo_user_known",
            "geo_country_match",
            "geo_city_match",
            "geo_distance_log_km",
            "geo_distance_under_50km",
            "geo_distance_under_250km",
            "geo_distance_under_1000km",
            "event_city_known",
            "event_country_known",
            "log_event_city_count",
            "log_event_country_count",
            "creator_is_friend",
            "creator_is_user",
            "log_creator_event_count",
            "log_event_yes",
            "log_event_maybe",
            "log_event_invited",
            "log_event_no",
            "log_event_total",
            "event_yes_ratio",
            "event_no_ratio",
            "event_net_positive",
            "event_yes_no_ratio",
            "event_label_rows",
            "event_label_pos_count",
            "event_label_neg_count",
            "event_label_interest_rate",
            "event_label_no_rate",
            "event_label_net_positive",
            "friend_yes",
            "friend_maybe",
            "friend_invited",
            "friend_no",
            "friend_yes_ratio",
            "friend_yes_event_yes_ratio",
            "friend_maybe_event_maybe_ratio",
            "friend_no_event_no_ratio",
            "friend_response_total",
            "friend_response_ratio",
            "friend_any_yes",
            "friend_any_no",
            "friend_net_positive",
            "log_user_friend_count",
            "user_rows",
            "user_pos_count",
            "user_neg_count",
            "user_interest_rate",
            "user_no_rate",
            "user_invited_rate",
            "user_is_cold",
            "user_age",
            "gender_male",
            "gender_female",
            "timezone_hours",
            "log_event_word_sum",
            "event_word_peak_ratio",
            "event_active_word_count",
            "event_word_density",
            "content_pos_cosine",
            "content_neg_cosine",
            "content_pos_minus_neg",
            "friend_content_pos_cosine",
            "friend_content_neg_cosine",
            "friend_content_pos_minus_neg",
            "friend_profile_pos_count",
            "friend_profile_neg_count",
            "friend_ppr_pos_score",
            "friend_ppr_neg_score",
            "friend_ppr_net_score",
            "log_friend_ppr_pos_score",
            "log_friend_ppr_neg_score",
        ]

    def transform_row(self, row: dict[str, str]) -> list[float]:
        user = row["user"]
        event = row["event"]
        invited = parse_int(row.get("invited", ""), 0)
        timestamp = parse_dt(row.get("timestamp", ""))

        user_info = self.users.get(user, {})
        ev = self.events.get(event, {})
        event_start = ev.get("start")
        if timestamp and event_start:
            if timestamp.tzinfo is None:
                timestamp = timestamp.replace(tzinfo=timezone.utc)
            if event_start.tzinfo is None:
                event_start = event_start.replace(tzinfo=timezone.utc)
            time_to_event_days = (event_start - timestamp).total_seconds() / 86400.0
        else:
            time_to_event_days = 0.0

        positive_time = max(time_to_event_days, 0.0)
        notify_hour = float(timestamp.hour) if timestamp else 0.0
        event_weekend = 1.0 if event_start and event_start.weekday() >= 5 else 0.0
        event_month = float(event_start.month) if event_start else 0.0
        location = user_info.get("location", "")
        city = ev.get("city", "")
        country = ev.get("country", "")
        same_city = 1.0 if city and city in location else 0.0
        same_country = 1.0 if country and country in location else 0.0
        user_geo = self.user_geo.get(user, {})
        event_country_code = match_country(country, self.geo) if self.geo else ""
        event_city_norm = normalize_geo_text(city)
        geo_user_known = 1.0 if user_geo.get("country") or user_geo.get("city") else 0.0
        geo_country_match = (
            1.0 if event_country_code and user_geo.get("country") and event_country_code == user_geo.get("country") else 0.0
        )
        geo_city_match = (
            1.0
            if event_city_norm and user_geo.get("city") and event_city_norm == user_geo.get("city")
            else 0.0
        )
        distance_km = haversine_km(
            float(user_geo.get("lat", 0.0) or 0.0),
            float(user_geo.get("lng", 0.0) or 0.0),
            float(ev.get("lat", 0.0) or 0.0),
            float(ev.get("lng", 0.0) or 0.0),
        )
        fset = self.friends.get(user, set())
        creator = ev.get("creator", "")
        creator_is_friend = 1.0 if creator and creator in fset else 0.0
        creator_is_user = 1.0 if creator and creator == user else 0.0
        creator_event_count = self.creator_counts.get(creator, 0)
        event_city_count = self.event_city_counts.get(city, 0)
        event_country_count = self.event_country_counts.get(country, 0)

        estats = self.event_stats.get(event, {})
        yes = estats.get("yes", 0)
        maybe = estats.get("maybe", 0)
        att_invited = estats.get("att_invited", 0)
        no = estats.get("no", 0)
        event_total = yes + maybe + att_invited + no
        yes_ratio = estats.get("yes_ratio", 0.0)
        no_ratio = estats.get("no_ratio", 0.0)
        event_net_positive = (yes + 0.5 * maybe - no) / event_total if event_total else 0.0
        event_yes_no_ratio = math.log1p(yes) - math.log1p(no)
        label_stats = self.event_label_stats.get(event, Counter())
        label_rows = label_stats.get("rows", 0)
        label_pos = label_stats.get("interested", 0)
        label_neg = label_stats.get("not_interested", 0)
        if (user, event) in self.label_history_pairs:
            label_rows -= 1
            label_pos -= parse_int(row.get("interested", "0"), 0)
            label_neg -= parse_int(row.get("not_interested", "0"), 0)
        label_rows = max(label_rows, 0)
        label_pos = max(label_pos, 0)
        label_neg = max(label_neg, 0)
        event_label_interest_rate = label_pos / label_rows if label_rows else 0.0
        event_label_no_rate = label_neg / label_rows if label_rows else 0.0
        event_label_net_positive = (label_pos - label_neg) / label_rows if label_rows else 0.0

        pf = self.pair_friend_stats.get((user, event), {})
        friend_yes = pf.get("friend_yes", 0)
        friend_maybe = pf.get("friend_maybe", 0)
        friend_invited = pf.get("friend_invited", 0)
        friend_no = pf.get("friend_no", 0)
        friend_count = self.friend_counts.get(user, 0)
        friend_yes_ratio = friend_yes / friend_count if friend_count else 0.0
        friend_yes_event_yes_ratio = friend_yes / yes if yes else 0.0
        friend_maybe_event_maybe_ratio = friend_maybe / maybe if maybe else 0.0
        friend_no_event_no_ratio = friend_no / no if no else 0.0
        friend_response_total = friend_yes + friend_maybe + friend_invited + friend_no
        friend_response_ratio = friend_response_total / friend_count if friend_count else 0.0
        friend_net_positive = friend_yes + 0.5 * friend_maybe - friend_no

        ustats = self.user_stats.get(user, Counter())
        rows = ustats.get("rows", 0)
        user_pos_count = ustats.get("interested", 0)
        user_neg_count = ustats.get("not_interested", 0)
        user_interest_rate = ustats.get("interested", 0) / rows if rows else 0.0
        user_no_rate = ustats.get("not_interested", 0) / rows if rows else 0.0
        user_invited_rate = ustats.get("invited", 0) / rows if rows else 0.0

        gender = user_info.get("gender", "")
        words = ev.get("words", [0] * 100)
        word_sum = ev.get("word_sum", 0)
        word_max = ev.get("word_max", 0)
        peak_ratio = word_max / word_sum if word_sum else 0.0
        active_word_count = sum(1 for value in words if value > 0)
        word_density = word_sum / active_word_count if active_word_count else 0.0
        content_pos = cosine_to_profile(words, self.pos_word_sum[user], self.pos_word_count[user])
        content_neg = cosine_to_profile(words, self.neg_word_sum[user], self.neg_word_count[user])
        friend_content_pos = cosine_to_profile(
            words, self.friend_pos_word_sum[user], self.friend_pos_word_count[user]
        )
        friend_content_neg = cosine_to_profile(
            words, self.friend_neg_word_sum[user], self.friend_neg_word_count[user]
        )
        friend_ppr_pos = self.friend_pos_event_scores[user].get(event, 0.0)
        friend_ppr_neg = self.friend_neg_event_scores[user].get(event, 0.0)

        return [
            float(invited),
            time_to_event_days,
            math.log1p(positive_time),
            1.0 if 0.0 <= time_to_event_days <= 1.0 else 0.0,
            1.0 if 0.0 <= time_to_event_days <= 7.0 else 0.0,
            1.0 if 0.0 <= time_to_event_days <= 30.0 else 0.0,
            event_weekend,
            event_month,
            notify_hour,
            float(ev.get("has_location", 0)),
            same_city,
            same_country,
            geo_user_known,
            geo_country_match,
            geo_city_match,
            math.log1p(distance_km) if distance_km else 0.0,
            1.0 if 0.0 < distance_km <= 50.0 else 0.0,
            1.0 if 0.0 < distance_km <= 250.0 else 0.0,
            1.0 if 0.0 < distance_km <= 1000.0 else 0.0,
            1.0 if city else 0.0,
            1.0 if country else 0.0,
            math.log1p(event_city_count),
            math.log1p(event_country_count),
            creator_is_friend,
            creator_is_user,
            math.log1p(creator_event_count),
            math.log1p(yes),
            math.log1p(maybe),
            math.log1p(att_invited),
            math.log1p(no),
            math.log1p(event_total),
            yes_ratio,
            no_ratio,
            event_net_positive,
            event_yes_no_ratio,
            math.log1p(label_rows),
            math.log1p(label_pos),
            math.log1p(label_neg),
            event_label_interest_rate,
            event_label_no_rate,
            event_label_net_positive,
            math.log1p(friend_yes),
            math.log1p(friend_maybe),
            math.log1p(friend_invited),
            math.log1p(friend_no),
            friend_yes_ratio,
            friend_yes_event_yes_ratio,
            friend_maybe_event_maybe_ratio,
            friend_no_event_no_ratio,
            math.log1p(friend_response_total),
            friend_response_ratio,
            1.0 if friend_yes > 0 else 0.0,
            1.0 if friend_no > 0 else 0.0,
            friend_net_positive,
            math.log1p(friend_count),
            math.log1p(rows),
            math.log1p(user_pos_count),
            math.log1p(user_neg_count),
            user_interest_rate,
            user_no_rate,
            user_invited_rate,
            1.0 if rows == 0 else 0.0,
            float(user_info.get("age_2012", 0)),
            1.0 if gender == "male" else 0.0,
            1.0 if gender == "female" else 0.0,
            float(user_info.get("timezone", 0)) / 60.0,
            math.log1p(word_sum),
            peak_ratio,
            float(active_word_count),
            word_density,
            content_pos,
            content_neg,
            content_pos - content_neg,
            friend_content_pos,
            friend_content_neg,
            friend_content_pos - friend_content_neg,
            math.log1p(self.friend_pos_word_count[user]),
            math.log1p(self.friend_neg_word_count[user]),
            friend_ppr_pos,
            friend_ppr_neg,
            friend_ppr_pos - friend_ppr_neg,
            math.log1p(friend_ppr_pos),
            math.log1p(friend_ppr_neg),
        ]


class Standardizer:
    def fit(self, rows: list[list[float]]):
        n = len(rows)
        m = len(rows[0])
        self.mean = [0.0] * m
        self.std = [0.0] * m
        for row in rows:
            for i, value in enumerate(row):
                self.mean[i] += value
        self.mean = [value / n for value in self.mean]
        for row in rows:
            for i, value in enumerate(row):
                diff = value - self.mean[i]
                self.std[i] += diff * diff
        self.std = [math.sqrt(value / n) or 1.0 for value in self.std]
        return self

    def transform(self, row: list[float]) -> list[float]:
        return [(value - self.mean[i]) / self.std[i] for i, value in enumerate(row)]


class LogisticSGD:
    def __init__(self, n_features: int, lr: float = 0.035, reg: float = 0.0005, epochs: int = 80):
        self.weights = [0.0] * (n_features + 1)
        self.lr = lr
        self.reg = reg
        self.epochs = epochs

    @staticmethod
    def sigmoid(x: float) -> float:
        if x < -35:
            return 0.0
        if x > 35:
            return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def predict_proba(self, row: list[float]) -> float:
        z = self.weights[0]
        for i, value in enumerate(row, start=1):
            z += self.weights[i] * value
        return self.sigmoid(z)

    def fit(self, rows: list[list[float]], labels: list[int], seed: int = 13):
        rng = random.Random(seed)
        order = list(range(len(rows)))
        pos = sum(labels)
        neg = len(labels) - pos
        pos_weight = neg / pos if pos else 1.0
        for epoch in range(self.epochs):
            rng.shuffle(order)
            rate = self.lr / (1.0 + 0.03 * epoch)
            for idx in order:
                x = rows[idx]
                y = labels[idx]
                p = self.predict_proba(x)
                sample_weight = pos_weight if y else 1.0
                grad = (p - y) * sample_weight
                self.weights[0] -= rate * grad
                for j, value in enumerate(x, start=1):
                    self.weights[j] -= rate * (grad * value + self.reg * self.weights[j])
        return self


def map_at_k(rows: list[tuple[str, str, int, float]], k: int = K) -> float:
    by_user = defaultdict(list)
    for user, event, label, score in rows:
        by_user[user].append((score, label, event))
    total = 0.0
    for values in by_user.values():
        positives = sum(label for _, label, _ in values)
        if positives == 0:
            continue
        hits = 0
        ap = 0.0
        for rank, (_, label, _) in enumerate(sorted(values, reverse=True)[:k], start=1):
            if label:
                hits += 1
                ap += hits / rank
        total += ap / min(positives, k)
    return total / len(by_user) if by_user else 0.0


def write_submission(path: Path, scored_rows: list[tuple[str, str, float]], legacy: bool = False):
    by_user = defaultdict(dict)
    for user, event, score in scored_rows:
        by_user[user][event] = max(score, by_user[user].get(event, -1.0))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["User", "Events"])
        for user in sorted(by_user, key=lambda x: int(x)):
            events = [event for event, _ in sorted(by_user[user].items(), key=lambda item: item[1], reverse=True)]
            if legacy:
                events_value = "[" + ", ".join(f"{event}L" for event in events) + "]"
            else:
                events_value = " ".join(events)
            writer.writerow([user, events_value])


def build_everything(include_test_metadata: bool = True):
    print("Reading train/test...")
    train_rows = read_interactions(DATA_DIR / "train.csv", has_labels=True)
    test_rows = read_interactions(DATA_DIR / "test.csv", has_labels=False)
    metadata_rows = train_rows + test_rows if include_test_metadata else train_rows
    candidate_users, candidate_events = collect_candidate_sets(metadata_rows, [])
    candidate_pairs = {(row["user"], row["event"]) for row in metadata_rows}

    print(f"Candidate users={len(candidate_users)} events={len(candidate_events)} pairs={len(candidate_pairs)}")
    print("Loading users...")
    users = load_users(DATA_DIR / "users.csv", candidate_users)
    print("Loading friends...")
    friends, friend_counts = load_friends(DATA_DIR / "user_friends.csv.gz", candidate_users)
    print("Loading event metadata...")
    events = load_events(DATA_DIR / "events.csv.gz", candidate_events)
    print("Loading event attendees and friend-event intersections...")
    event_stats, pair_friend_stats = load_event_attendees(
        DATA_DIR / "event_attendees.csv.gz", candidate_events, friends, candidate_pairs
    )
    print("Building user history...")
    user_history = build_user_history(train_rows, events)
    friend_profiles = build_friend_profiles(
        friends,
        user_history[1],
        user_history[2],
        user_history[3],
        user_history[4],
        user_history[5],
        user_history[6],
    )
    builder = FeatureBuilder(
        users,
        friends,
        friend_counts,
        events,
        event_stats,
        pair_friend_stats,
        *user_history,
        *friend_profiles,
    )
    return train_rows, test_rows, builder


def run(args):
    train_rows, test_rows, builder = build_everything()
    users = sorted({row["user"] for row in train_rows}, key=lambda x: int(x))
    valid_users = set(users[int(len(users) * 0.8) :])
    train_fit = [row for row in train_rows if row["user"] not in valid_users]
    valid = [row for row in train_rows if row["user"] in valid_users]

    print("Vectorizing...")
    valid_history = build_user_history(train_fit, builder.events)
    valid_builder = FeatureBuilder(
        builder.users,
        builder.friends,
        builder.friend_counts,
        builder.events,
        builder.event_stats,
        builder.pair_friend_stats,
        *valid_history,
    )

    x_fit_raw = [valid_builder.transform_row(row) for row in train_fit]
    y_fit = [parse_int(row["interested"], 0) for row in train_fit]
    x_valid_raw = [valid_builder.transform_row(row) for row in valid]
    y_valid = [parse_int(row["interested"], 0) for row in valid]

    scaler = Standardizer().fit(x_fit_raw)
    x_fit = [scaler.transform(row) for row in x_fit_raw]
    x_valid = [scaler.transform(row) for row in x_valid_raw]

    print(f"Training logistic SGD on {len(x_fit)} rows, validating on {len(x_valid)} rows...")
    model = LogisticSGD(len(builder.feature_names), epochs=args.epochs, lr=args.lr, reg=args.reg).fit(x_fit, y_fit)
    valid_scored = []
    for row, x, label in zip(valid, x_valid, y_valid):
        valid_scored.append((row["user"], row["event"], label, model.predict_proba(x)))
    score = map_at_k(valid_scored)
    print(f"Validation MAP@200: {score:.6f}")

    print("Retraining on full train...")
    x_all_raw = [builder.transform_row(row) for row in train_rows]
    y_all = [parse_int(row["interested"], 0) for row in train_rows]
    full_scaler = Standardizer().fit(x_all_raw)
    x_all = [full_scaler.transform(row) for row in x_all_raw]
    full_model = LogisticSGD(len(builder.feature_names), epochs=args.epochs, lr=args.lr, reg=args.reg).fit(x_all, y_all)

    print("Scoring test and writing submission...")
    scored_test = []
    for row in test_rows:
        x = full_scaler.transform(builder.transform_row(row))
        scored_test.append((row["user"], row["event"], full_model.predict_proba(x)))
    write_submission(DATA_DIR / args.output, scored_test)
    legacy_output = DATA_DIR / args.output.replace(".csv", "_legacy.csv")
    write_submission(legacy_output, scored_test, legacy=True)
    print(f"Wrote {args.output} with {len({r['user'] for r in test_rows})} users and {len(test_rows)} scored pairs.")
    print(f"Wrote {legacy_output.name} in official benchmark-style list format.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="submission_baseline.csv")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.035)
    parser.add_argument("--reg", type=float, default=0.0005)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
