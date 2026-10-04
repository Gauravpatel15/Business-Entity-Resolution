"""Compact, local-only business entity resolution pipeline.

Commands:
  python pipeline.py run --train-dir DATA/train --test-dir DATA/test --output-dir output --work-dir work
  python pipeline.py predict --test-dir DATA/test --output-dir output --work-dir work
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import re
import sqlite3
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from lightgbm import LGBMClassifier, Booster
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein
from text_unidecode import unidecode


LEGAL_WORDS = {
    "inc", "incorporated", "corp", "corporation", "co", "company", "ltd", "limited",
    "llc", "llp", "pllc", "pvt", "private", "sa", "sas", "sasu", "sarl", "sci",
    "eurl", "snc", "gmbh", "bv", "group", "holding", "holdings", "enterprise",
    "enterprises", "services", "solutions", "technology", "technologies", "trading",
}
ARTICLES = {"the", "a", "an", "le", "la", "les", "l", "un", "une", "de", "des", "du", "d"}
ADDRESS_EXPANSIONS = {
    "rd": "road", "ave": "avenue", "blvd": "boulevard", "dr": "drive", "ln": "lane",
    "ct": "court", "hwy": "highway", "pkwy": "parkway", "st": "street", "ste": "suite",
    "apt": "apartment", "bldg": "building", "r": "rue", "av": "avenue", "bd": "boulevard",
    "ch": "chemin", "rte": "route", "imp": "impasse", "all": "allee", "bat": "batiment",
}
ADDRESS_STOP = set(ADDRESS_EXPANSIONS.values()) | {
    "street", "road", "avenue", "boulevard", "drive", "lane", "court", "highway", "suite",
    "apartment", "building", "floor", "unit", "near", "opposite", "block", "sector", "phase",
    "india", "france", "usa", "us", "state", "city", "nagar", "colony", "rue", "chemin",
    "route", "impasse", "allee", "residence", "zone", "industrielle", "de", "des", "du", "la", "le",
}
NON_ALNUM = re.compile(r"[^a-z0-9\s]")
SPACES = re.compile(r"\s+")
DIGIT_WORD = re.compile(r"(\d+)([a-z]+)|([a-z]+)(\d+)")


@dataclass(frozen=True)
class Entity:
    entity_id: str
    name: str
    address: str
    country: str


@dataclass(frozen=True)
class Normalized:
    clean_name: str
    core_name: str
    clean_address: str
    numbers: frozenset[str]
    primary_number: str


FEATURE_NAMES = [
    "name_exact", "core_exact", "name_lev", "name_jw", "name_sort", "name_set", "name_jaccard",
    "name_overlap", "name_length_ratio", "name_3gram", "first_token_similarity", "address_present",
    "address_exact", "address_lev", "address_jw", "address_sort", "address_set", "address_jaccard",
    "number_overlap", "number_conflict", "primary_number_match", "primary_number_conflict", "name_address_product",
    "exact_core_missing_address", "shared_block_score", "is_source2",
]


def stable_number(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest(), "big")


def clean_text(value: str) -> str:
    value = "" if value is None else str(value)
    if value.lower().strip() in {"", "null", "none", "nan"}:
        return ""
    value = unicodedata.normalize("NFKC", unidecode(value)).lower()
    value = re.sub(r"https?://(?:www\.)?", "", value)
    value = re.sub(r"\.(?:com|org|net|in|co|io|fr)\b", " ", value)
    value = DIGIT_WORD.sub(lambda m: " ".join(x for x in m.groups() if x), value)
    return SPACES.sub(" ", NON_ALNUM.sub(" ", value)).strip()


def normalize(entity: Entity) -> Normalized:
    name = clean_text(entity.name)
    name_tokens = name.split()
    core_tokens = [token for token in name_tokens if token not in LEGAL_WORDS]
    if len(core_tokens) > 1 and core_tokens[0] in ARTICLES:
        core_tokens = core_tokens[1:]
    core = " ".join(core_tokens or name_tokens)
    raw_address = clean_text(entity.address)
    address_tokens = [ADDRESS_EXPANSIONS.get(token, token) for token in raw_address.split()]
    address = " ".join(address_tokens)
    ordered_numbers = [str(int(number)) for number in re.findall(r"\b\d+\b", raw_address)]
    numbers = frozenset(ordered_numbers)
    primary = ordered_numbers[0] if ordered_numbers else ""
    return Normalized(name, core, address, numbers, primary)


def read_tsv(path: Path) -> Iterable[Entity]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            yield Entity(
                row.get("entity_id", "").strip(), row.get("business_name", ""),
                row.get("business_address", ""), row.get("country", "").strip(),
            )


def read_ground_truth(path: Path) -> dict[str, set[str]]:
    answer: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            answer[row["source1_entity_id"]] = set(filter(None, row.get("matched_entity_ids", "").split(",")))
    return answer


def significant(tokens: list[str]) -> list[str]:
    return [token for token in tokens if len(token) >= 3 and token not in ADDRESS_STOP and not token.isdigit()]


def blocking_keys(value: Normalized) -> set[tuple[str, str]]:
    """High-recall keys. Country is an outer partition, never a hard-coded category."""
    keys: set[tuple[str, str]] = set()
    names, addresses = value.core_name.split(), value.clean_address.split()
    compact = "".join(names)
    if len(compact) >= 4:
        keys.add(("name_compact", compact))
        keys.add(("name_sorted_compact", "".join(sorted(names))))
    if len(value.core_name) >= 4:
        keys.add(("name_core", value.core_name))
    if len(names) >= 2:
        keys.add(("name_first_two", "_".join(names[:2])))
        keys.add(("name_sorted", " ".join(sorted(names))))
    for token in significant(names):
        keys.add(("name_token", token))

    rare_address = significant(addresses)
    for number in value.numbers:
        for token in rare_address[:5]:
            keys.add(("number_address", f"{number}_{token}"))
        if addresses:
            keys.add(("number_tail", f"{number}_{addresses[-1]}"))
        if names:
            keys.add(("name_number", f"{names[0]}_{number}"))
    if names:
        for token in addresses[-3:]:
            if len(token) >= 3:
                keys.add(("name_location", f"{names[0]}_{token}"))
    for left, right in zip(rare_address[:4], rare_address[1:5]):
        keys.add(("address_pair", "_".join(sorted((left, right)))))
    return keys


def build_index(targets: dict[str, Normalized]) -> dict[tuple[str, str], list[str]]:
    index: dict[tuple[str, str], list[str]] = defaultdict(list)
    for target_id, value in targets.items():
        for key in blocking_keys(value):
            index[key].append(target_id)
    return index


def candidates(value: Normalized, index: dict[tuple[str, str], list[str]], top_k: int) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for key in blocking_keys(value):
        posting = index.get(key, [])
        # Very broad one-token postings are noise. Composite and exact-name keys remain usable.
        if key[0] in {"name_token", "name_location"} and len(posting) > 5000:
            continue
        if key[0] == "address_pair" and len(posting) > 2500:
            continue
        weight = 1.0 / math.log2(2.0 + len(posting))
        for target_id in posting:
            scores[target_id] += weight
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]


def grams(value: str) -> set[str]:
    return {value[i:i + 3] for i in range(max(1, len(value) - 2))} if value else set()


def jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def pair_features(left: Normalized, right: Normalized, target_id: str, block_score: float) -> list[float]:
    name_a, name_b = left.clean_name, right.clean_name
    core_a, core_b = left.core_name, right.core_name
    name_tokens_a, name_tokens_b = set(core_a.split()), set(core_b.split())
    name_overlap = len(name_tokens_a & name_tokens_b) / min(len(name_tokens_a), len(name_tokens_b)) if name_tokens_a and name_tokens_b else 0.0
    first = Levenshtein.normalized_similarity(core_a.split()[0], core_b.split()[0]) if core_a and core_b else 0.0
    address_present = float(bool(right.clean_address))
    if left.clean_address and right.clean_address:
        address_tokens_a, address_tokens_b = set(left.clean_address.split()), set(right.clean_address.split())
        address_exact = float(left.clean_address == right.clean_address)
        address_lev = Levenshtein.normalized_similarity(left.clean_address, right.clean_address)
        address_jw = JaroWinkler.similarity(left.clean_address, right.clean_address)
        address_sort = fuzz.token_sort_ratio(left.clean_address, right.clean_address) / 100.0
        address_set = fuzz.token_set_ratio(left.clean_address, right.clean_address) / 100.0
        address_jaccard = jaccard(address_tokens_a, address_tokens_b)
    else:
        address_exact = address_lev = address_jw = address_sort = address_set = address_jaccard = 0.0
    shared_numbers = bool(left.numbers & right.numbers)
    number_conflict = bool(left.numbers and right.numbers and not shared_numbers)
    primary_match = float(bool(left.primary_number and left.primary_number == right.primary_number))
    primary_conflict = float(bool(left.primary_number and right.primary_number and left.primary_number != right.primary_number))
    name_sort = fuzz.token_sort_ratio(name_a, name_b) / 100.0 if name_a and name_b else 0.0
    return [
        float(bool(name_a and name_a == name_b)), float(bool(core_a and core_a == core_b)),
        Levenshtein.normalized_similarity(name_a, name_b) if name_a and name_b else 0.0,
        JaroWinkler.similarity(name_a, name_b) if name_a and name_b else 0.0,
        name_sort, fuzz.token_set_ratio(name_a, name_b) / 100.0 if name_a and name_b else 0.0,
        jaccard(name_tokens_a, name_tokens_b), name_overlap,
        min(len(name_a), len(name_b)) / max(len(name_a), len(name_b)) if name_a and name_b else 0.0,
        jaccard(grams(core_a), grams(core_b)), first, address_present, address_exact, address_lev, address_jw,
        address_sort, address_set, address_jaccard, float(shared_numbers), float(number_conflict), primary_match,
        primary_conflict, name_sort * (address_sort if address_present else name_sort),
        float(bool(core_a and core_a == core_b and not right.clean_address)), block_score,
        float(target_id.startswith("S2-")),
    ]


def score_entity(true_ids: set[str], predicted_ids: set[str]) -> float:
    if not true_ids:
        return float(not predicted_ids)
    if not predicted_ids:
        return 0.0
    true_positive = len(true_ids & predicted_ids)
    if not true_positive:
        return 0.0
    precision, recall = true_positive / len(predicted_ids), true_positive / len(true_ids)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def metric(truth: dict[str, set[str]], predicted: dict[str, set[str]]) -> dict[str, float]:
    f05 = [score_entity(answer, predicted.get(source_id, set())) for source_id, answer in truth.items()]
    tp = sum(len(answer & predicted.get(source_id, set())) for source_id, answer in truth.items())
    total_predicted = sum(len(predicted.get(source_id, set())) for source_id in truth)
    total_true = sum(len(answer) for answer in truth.values())
    singleton_ids = [source_id for source_id, answer in truth.items() if not answer]
    return {
        "macro_f05": float(np.mean(f05)) if f05 else 0.0,
        "precision": tp / total_predicted if total_predicted else 1.0,
        "recall": tp / total_true if total_true else 0.0,
        "singleton_accuracy": sum(not predicted.get(source_id, set()) for source_id in singleton_ids) / len(singleton_ids) if singleton_ids else 1.0,
    }


def select_predictions(scores: dict[str, list[tuple[str, float]]], threshold_s2: float, threshold_s3: float, deduplicate: bool) -> dict[str, set[str]]:
    kept = [
        (probability, source_id, target_id)
        for source_id, rows in scores.items()
        for target_id, probability in rows
        if probability >= (threshold_s2 if target_id.startswith("S2-") else threshold_s3)
    ]
    result: dict[str, set[str]] = {source_id: set() for source_id in scores}
    if deduplicate:
        used: set[str] = set()
        for probability, source_id, target_id in sorted(kept, reverse=True):
            if target_id not in used:
                used.add(target_id)
                result[source_id].add(target_id)
    else:
        for _, source_id, target_id in kept:
            result[source_id].add(target_id)
    return result


def tune_thresholds(truth: dict[str, set[str]], scores: dict[str, list[tuple[str, float]]], deduplicate: bool) -> tuple[float, float, dict[str, float]]:
    grid = np.arange(0.40, 0.951, 0.05)
    best = (0.70, 0.75, {"macro_f05": -1.0})
    for s2 in grid:
        for s3 in grid:
            value = metric(truth, select_predictions(scores, float(s2), float(s3), deduplicate))
            if value["macro_f05"] > best[2]["macro_f05"]:
                best = (float(s2), float(s3), value)
    return best


def targets_for_country(source2: Path, source3: Path, country: str) -> dict[str, Normalized]:
    targets: dict[str, Normalized] = {}
    for source in (source2, source3):
        for entity in read_tsv(source):
            if entity.country == country:
                targets[entity.entity_id] = normalize(entity)
    return targets


def group_source1(path: Path, wanted: set[str] | None = None) -> dict[str, list[Entity]]:
    groups: dict[str, list[Entity]] = defaultdict(list)
    for entity in read_tsv(path):
        if wanted is None or entity.entity_id in wanted:
            groups[entity.country].append(entity)
    return groups


def create_training_pairs(train_dir: Path, training: dict[str, Entity], top_k: int, negative_limit: int) -> tuple[np.ndarray, np.ndarray, float]:
    truth = read_ground_truth(train_dir / "train_ground_truth.tsv")
    rows: list[list[float]] = []
    labels: list[int] = []
    retrieved_true, total_true = 0, 0
    by_country: dict[str, list[Entity]] = defaultdict(list)
    for entity in training.values():
        by_country[entity.country].append(entity)
    for country, source1 in by_country.items():
        targets = targets_for_country(train_dir / "train_source2.tsv", train_dir / "train_source3.tsv", country)
        index = build_index(targets)
        for entity in source1:
            value, actual = normalize(entity), truth.get(entity.entity_id, set())
            candidate_rows = candidates(value, index, top_k)
            candidate_ids = {target_id for target_id, _ in candidate_rows}
            retrieved_true += len(actual & candidate_ids)
            total_true += len(actual)
            # All positives are included for class learning, even when blocking missed them.
            for target_id in actual:
                if target_id in targets:
                    block_score = dict(candidate_rows).get(target_id, 0.0)
                    rows.append(pair_features(value, targets[target_id], target_id, block_score))
                    labels.append(1)
            negatives = 0
            for target_id, block_score in candidate_rows:
                if target_id not in actual:
                    rows.append(pair_features(value, targets[target_id], target_id, block_score))
                    labels.append(0)
                    negatives += 1
                    if negatives >= max(negative_limit, 3 * len(actual)):
                        break
    return np.asarray(rows, dtype=np.float32), np.asarray(labels, dtype=np.int8), retrieved_true / total_true if total_true else 1.0


def validation_scores(train_dir: Path, validation: dict[str, Entity], model: LGBMClassifier, top_k: int) -> tuple[dict[str, list[tuple[str, float]]], float]:
    truth = read_ground_truth(train_dir / "train_ground_truth.tsv")
    scored: dict[str, list[tuple[str, float]]] = {source_id: [] for source_id in validation}
    got, possible = 0, 0
    by_country: dict[str, list[Entity]] = defaultdict(list)
    for entity in validation.values():
        by_country[entity.country].append(entity)
    for country, source1 in by_country.items():
        targets = targets_for_country(train_dir / "train_source2.tsv", train_dir / "train_source3.tsv", country)
        index = build_index(targets)
        feature_rows, labels = [], []
        for entity in source1:
            value = normalize(entity)
            candidate_rows = candidates(value, index, top_k)
            labels.append((entity.entity_id, [target_id for target_id, _ in candidate_rows]))
            got += len(truth[entity.entity_id] & set(labels[-1][1]))
            possible += len(truth[entity.entity_id])
            feature_rows.extend(pair_features(value, targets[target_id], target_id, block_score) for target_id, block_score in candidate_rows)
        probabilities = model.predict_proba(np.asarray(feature_rows, dtype=np.float32))[:, 1] if feature_rows else []
        cursor = 0
        for source_id, target_ids in labels:
            count = len(target_ids)
            scored[source_id] = list(zip(target_ids, map(float, probabilities[cursor:cursor + count])))
            cursor += count
    return scored, got / possible if possible else 1.0


def split_training_entities(train_dir: Path, validation_fraction: float, train_limit: int, validation_limit: int) -> tuple[dict[str, Entity], dict[str, Entity], dict[str, set[str]], bool]:
    truth = read_ground_truth(train_dir / "train_ground_truth.tsv")
    validation, training = {}, {}
    for entity in read_tsv(train_dir / "train_source1.tsv"):
        if entity.entity_id not in truth:
            continue
        if stable_number(entity.entity_id) % 10_000 < int(validation_fraction * 10_000):
            validation[entity.entity_id] = entity
        else:
            training[entity.entity_id] = entity
    ordered = sorted(training, key=stable_number)[:train_limit]
    training = {source_id: training[source_id] for source_id in ordered}
    validation = {source_id: validation[source_id] for source_id in sorted(validation, key=stable_number)[:validation_limit]}
    ownership: dict[str, str] = {}
    exclusive = True
    for source_id, target_ids in truth.items():
        for target_id in target_ids:
            if target_id in ownership and ownership[target_id] != source_id:
                exclusive = False
            ownership[target_id] = source_id
    return training, validation, truth, exclusive


def train(args: argparse.Namespace) -> dict:
    train_dir, work_dir = Path(args.train_dir), Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    training, validation, truth, exclusive = split_training_entities(
        train_dir, args.validation_fraction, args.train_limit, args.validation_limit
    )
    if not training or not validation:
        raise ValueError("The deterministic train/validation split is empty; check the training TSV files.")
    X, y, training_candidate_recall = create_training_pairs(train_dir, training, args.top_k, args.negative_limit)
    if len(np.unique(y)) != 2:
        raise ValueError("Training pairs contain one class only; increase --train-limit or inspect the ground truth.")
    model = LGBMClassifier(
        objective="binary", n_estimators=650, learning_rate=0.035, num_leaves=63, max_depth=-1,
        min_child_samples=30, subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
        random_state=2026, n_jobs=args.n_jobs, verbosity=-1,
    ).fit(X, y)
    scores, validation_candidate_recall = validation_scores(train_dir, validation, model, args.top_k)
    validation_truth = {source_id: truth[source_id] for source_id in validation}
    threshold_s2, threshold_s3, metrics = tune_thresholds(validation_truth, scores, exclusive)
    model.booster_.save_model(str(work_dir / "model.txt"))
    metadata = {
        "threshold_s2": threshold_s2, "threshold_s3": threshold_s3, "deduplicate_targets": exclusive,
        "validation": metrics, "training_candidate_recall": training_candidate_recall,
        "validation_candidate_recall": validation_candidate_recall, "training_entities": len(training),
        "validation_entities": len(validation), "feature_names": FEATURE_NAMES, "top_k": args.top_k,
    }
    (work_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return metadata


def write_predictions(args: argparse.Namespace, metadata: dict) -> None:
    test_dir, output_dir, work_dir = Path(args.test_dir), Path(args.output_dir), Path(args.work_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    source1 = group_source1(test_dir / "test_source1.tsv")
    ordered_ids = [entity.entity_id for entities in source1.values() for entity in entities]
    # Preserve original input order, even though scoring is country-partitioned.
    ordered_ids = [entity.entity_id for entity in read_tsv(test_dir / "test_source1.tsv")]
    model = Booster(model_file=str(work_dir / "model.txt"))
    db_path = work_dir / "accepted_scores.sqlite"
    if db_path.exists():
        db_path.unlink()
    connection = sqlite3.connect(db_path)
    cursor = connection.cursor()
    cursor.execute("CREATE TABLE accepted (source_id TEXT, target_id TEXT, probability REAL)")
    cursor.execute("CREATE INDEX accepted_probability ON accepted(probability DESC)")
    threshold_s2, threshold_s3 = metadata["threshold_s2"], metadata["threshold_s3"]
    candidate_path = output_dir / "candidate_pairs.tsv"
    with candidate_path.open("w", encoding="utf-8", newline="") as candidate_file:
        writer = csv.writer(candidate_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "candidate_entity_ids"])
        for country, entities in source1.items():
            targets = targets_for_country(test_dir / "test_source2.tsv", test_dir / "test_source3.tsv", country)
            index = build_index(targets)
            accepted_count = 0
            for start in range(0, len(entities), args.prediction_batch_size):
                feature_rows, references = [], []
                for entity in entities[start:start + args.prediction_batch_size]:
                    value = normalize(entity)
                    rows = candidates(value, index, args.top_k)
                    writer.writerow([entity.entity_id, ",".join(target_id for target_id, _ in rows)])
                    feature_rows.extend(pair_features(value, targets[target_id], target_id, block_score) for target_id, block_score in rows)
                    references.extend((entity.entity_id, target_id) for target_id, _ in rows)
                probabilities = model.predict(np.asarray(feature_rows, dtype=np.float32)) if feature_rows else []
                passing = [
                    (source_id, target_id, float(probability))
                    for (source_id, target_id), probability in zip(references, probabilities)
                    if probability >= (threshold_s2 if target_id.startswith("S2-") else threshold_s3)
                ]
                cursor.executemany("INSERT INTO accepted VALUES (?, ?, ?)", passing)
                accepted_count += len(passing)
                connection.commit()
                print(f"Scored {country}: {min(start + args.prediction_batch_size, len(entities)):,}/{len(entities):,} source-1 entities")
            connection.commit()
            del targets, index
            gc.collect()
            print(f"Finished {country}: {accepted_count:,} links above threshold")
    cursor.execute("CREATE TABLE matches (source_id TEXT, target_id TEXT)")
    if metadata["deduplicate_targets"]:
        cursor.execute("CREATE TABLE used_targets (target_id TEXT PRIMARY KEY)")
        read_cursor, write_cursor = connection.cursor(), connection.cursor()
        for source_id, target_id, _ in read_cursor.execute("SELECT source_id, target_id, probability FROM accepted ORDER BY probability DESC"):
            if write_cursor.execute("INSERT OR IGNORE INTO used_targets VALUES (?)", (target_id,)).rowcount:
                write_cursor.execute("INSERT INTO matches VALUES (?, ?)", (source_id, target_id))
    else:
        cursor.execute("INSERT INTO matches SELECT source_id, target_id FROM accepted")
    connection.commit()
    cursor.execute("CREATE INDEX matches_source ON matches(source_id)")
    matching_path = output_dir / "matching_results.tsv"
    with matching_path.open("w", encoding="utf-8", newline="") as matching_file:
        writer = csv.writer(matching_file, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for source_id in ordered_ids:
            target_ids = [row[0] for row in cursor.execute("SELECT target_id FROM matches WHERE source_id = ? ORDER BY target_id", (source_id,))]
            writer.writerow([source_id, ",".join(target_ids)])
    connection.close()
    db_path.unlink(missing_ok=True)
    print(f"Wrote {matching_path} and {candidate_path}")


def run(args: argparse.Namespace) -> None:
    metadata = train(args)
    write_predictions(args, metadata)
    report = {"model": "LightGBM", "validation": metadata["validation"], "candidate_recall": metadata["validation_candidate_recall"]}
    (Path(args.work_dir) / "validation_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stochastic Pirates entity-resolution pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "predict"):
        child = subparsers.add_parser(command)
        child.add_argument("--test-dir", required=True)
        child.add_argument("--output-dir", required=True)
        child.add_argument("--work-dir", required=True)
        child.add_argument("--top-k", type=int, default=50)
        child.add_argument("--prediction-batch-size", type=int, default=5000)
    train_parser = subparsers.choices["run"]
    train_parser.add_argument("--train-dir", required=True)
    train_parser.add_argument("--train-limit", type=int, default=60000)
    train_parser.add_argument("--validation-fraction", type=float, default=0.15)
    train_parser.add_argument("--validation-limit", type=int, default=10000)
    train_parser.add_argument("--negative-limit", type=int, default=8)
    train_parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        metadata = json.loads((Path(args.work_dir) / "model_metadata.json").read_text(encoding="utf-8"))
        write_predictions(args, metadata)


if __name__ == "__main__":
    main()
