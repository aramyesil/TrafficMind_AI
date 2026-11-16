"""Audit the original YOLO dataset and write a reproducible grouped split.

The original files under ``train/`` are read only. Split images and labels are
copied into ``dataset_split/``. Randomness is controlled by RANDOM_SEED.

Leakage groups keep these images together:
- exact file duplicates (MD5)
- images with an identical dHash, including a mirrored copy
- neighboring filename frames whose original dHash distance is small
- every frame whose name contains an explicit ``_mp4-`` video prefix

A looser all-pairs similarity graph is intentionally not used. Traffic scenes
are similar enough that distance <= 2 already chained 1,478 images together,
and distance <= 10 chained 8,190. Filename stems such as ``DSC_0001`` are also
not treated as source IDs, because the same stem can contain different scenes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "train" / "images"
LABEL_DIR = PROJECT_ROOT / "train" / "labels"
SPLIT_ROOT = PROJECT_ROOT / "dataset_split"
REPORT_DIR = PROJECT_ROOT / "reports"
CACHE_HASHES = REPORT_DIR / "_cache_hashes.npy"
CACHE_META = REPORT_DIR / "_cache_meta.json"

CLASS_NAMES = ["bus", "car", "microbus", "motorbike", "pickup-van", "truck"]
RANDOM_SEED = 42
SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}
EXPECTED_SIZE = (640, 640)
IDENTICAL_HASH_MAX_DISTANCE = 0
SEQUENCE_MAX_DISTANCE = 12
SEQUENCE_MAX_FRAME_GAP = 3
TINY_BOX_AREA = (8 / 640) ** 2

SEQUENCE_PATTERNS = (
    re.compile(r"^(?P<prefix>DSC)_(?P<frame>\d+)_JPG$"),
    re.compile(r"^(?P<prefix>[A-Za-z][A-Za-z0-9]*)-(?P<frame>\d+)-_jpg$"),
    re.compile(r"^(?P<prefix>.+)_mp4-(?P<frame>\d+)_jpg$"),
)
VIDEO_PATTERN = re.compile(r"^(?P<prefix>.+)_mp4-\d+_jpg$")

POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def dhash(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    return np.packbits(diff.reshape(-1))


def hamming_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Hamming distance between rows of two uint8 arrays with shape (..., 8)."""
    xor = np.bitwise_xor(left[:, None, :], right[None, :, :])
    return POPCOUNT[xor].sum(axis=-1)


def original_distance(hashes: np.ndarray, left: int, right: int) -> int:
    xor = np.bitwise_xor(hashes[left, 0], hashes[right, 0])
    return int(POPCOUNT[xor].sum())


def source_stem(filename: str) -> str:
    return Path(filename).stem.split(".rf.", 1)[0]


def sequence_key(stem: str) -> tuple[str, int] | None:
    for pattern in SEQUENCE_PATTERNS:
        match = pattern.match(stem)
        if match:
            return match.group("prefix"), int(match.group("frame"))
    return None


def video_prefix(stem: str) -> str | None:
    match = VIDEO_PATTERN.match(stem)
    if match:
        return match.group("prefix") + "_mp4"
    return None


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def groups(self) -> dict[int, list[int]]:
        grouped: dict[int, list[int]] = defaultdict(list)
        for index in range(len(self.parent)):
            grouped[self.find(index)].append(index)
        return grouped


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    issues: list[str] = []
    boxes: list[tuple[int, float, float, float, float]] = []
    text = path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        issues.append("empty_label")
        return boxes, issues

    seen: set[str] = set()
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line in seen:
            issues.append(f"duplicate_line:{line_number}")
        seen.add(line)
        parts = line.split()
        if len(parts) != 5:
            issues.append(f"malformed_line:{line_number}")
            continue
        try:
            class_id = int(parts[0])
            cx, cy, width, height = (float(value) for value in parts[1:])
        except ValueError:
            issues.append(f"malformed_line:{line_number}")
            continue
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            issues.append(f"class_id:{line_number}:{class_id}")
            continue
        coordinates = (cx, cy, width, height)
        if not all(np.isfinite(value) for value in coordinates):
            issues.append(f"non_finite_box:{line_number}")
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
            issues.append(f"bbox_out_of_range:{line_number}")
            continue
        boxes.append((class_id, cx, cy, width, height))
    return boxes, issues


def load_records() -> tuple[list[dict], dict]:
    image_paths = sorted(IMAGE_DIR.glob("*.jpg"))
    label_paths = sorted(LABEL_DIR.glob("*.txt"))
    image_names = {path.name for path in image_paths}
    label_names = {path.name for path in label_paths}
    image_stems = {path.stem for path in image_paths}
    label_stems = {path.stem for path in label_paths}

    records: list[dict] = []
    for image_path in image_paths:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        boxes: list[tuple[int, float, float, float, float]] = []
        label_issues: list[str] = []
        if label_path.exists():
            boxes, label_issues = parse_label(label_path)
        else:
            label_issues = ["missing_label"]
        records.append(
            {
                "image_path": image_path,
                "label_path": label_path if label_path.exists() else None,
                "filename": image_path.name,
                "source_stem": source_stem(image_path.name),
                "boxes": boxes,
                "label_issues": label_issues,
            }
        )

    extra_labels = sorted(label_stems - image_stems)
    return records, {
        "image_count": len(image_paths),
        "label_count": len(label_paths),
        "image_extensions": dict(Counter(path.suffix.lower() for path in image_paths)),
        "label_extensions": dict(Counter(path.suffix.lower() for path in label_paths)),
        "images_missing_labels": sorted(image_stems - label_stems),
        "labels_missing_images": extra_labels,
        "unexpected_image_names": sorted(name for name in image_names if not name.lower().endswith(".jpg")),
    }


def scan_images(records: list[dict]) -> tuple[np.ndarray, dict]:
    if CACHE_HASHES.exists() and CACHE_META.exists():
        cached_meta = json.loads(CACHE_META.read_text(encoding="utf-8"))
        cached_names = cached_meta.get("filenames", [])
        current_names = [record["filename"] for record in records]
        if cached_names == current_names:
            print(f"Reusing image cache: {CACHE_HASHES.name}")
            hashes = np.load(CACHE_HASHES)
            for record, item in zip(records, cached_meta["items"]):
                record.update(item)
            return hashes, cached_meta["scan_summary"]

    hashes = np.zeros((len(records), 2, 8), dtype=np.uint8)
    items = []
    unreadable = []
    wrong_size = []
    md5_groups: dict[str, list[str]] = defaultdict(list)

    for index, record in enumerate(records):
        data = record["image_path"].read_bytes()
        digest = hashlib.md5(data).hexdigest()
        md5_groups[digest].append(record["filename"])
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            unreadable.append(record["filename"])
            item = {"md5": digest, "width": None, "height": None, "channels": None, "readable": False}
        else:
            height, width = image.shape[:2]
            channels = 1 if image.ndim == 2 else image.shape[2]
            if (width, height) != EXPECTED_SIZE or channels != 3:
                wrong_size.append({"filename": record["filename"], "width": width, "height": height, "channels": channels})
            hashes[index, 0] = dhash(image)
            hashes[index, 1] = dhash(cv2.flip(image, 1))
            item = {"md5": digest, "width": width, "height": height, "channels": channels, "readable": True}
        record.update(item)
        items.append({key: record[key] for key in ("md5", "width", "height", "channels", "readable")})
        if (index + 1) % 1000 == 0 or index + 1 == len(records):
            print(f"Scanned images: {index + 1}/{len(records)}")

    exact_duplicate_groups = [
        sorted(names) for names in md5_groups.values() if len(names) > 1
    ]
    scan_summary = {
        "unreadable_images": unreadable,
        "unexpected_dimensions": wrong_size[:50],
        "unexpected_dimension_count": len(wrong_size),
        "exact_duplicate_group_count": len(exact_duplicate_groups),
        "exact_duplicate_image_count": int(sum(len(group) for group in exact_duplicate_groups)),
        "exact_duplicate_examples": exact_duplicate_groups[:20],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(CACHE_HASHES, hashes)
    CACHE_META.write_text(
        json.dumps(
            {
                "filenames": [record["filename"] for record in records],
                "items": items,
                "scan_summary": scan_summary,
            }
        ),
        encoding="utf-8",
    )
    return hashes, scan_summary


def compare_hashes(hashes: np.ndarray, union: UnionFind) -> tuple[dict, int, int]:
    """Nearest-neighbor distances, identical-hash groups, and mirror groups."""
    count = hashes.shape[0]
    nearest = np.full(count, 64, dtype=np.uint16)
    identical_edges = 0
    mirror_edges = 0
    chunk_size = 256
    original = hashes[:, 0, :]
    flipped = hashes[:, 1, :]

    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        distance = hamming_matrix(original[start:stop], original)
        mirror_distance = hamming_matrix(original[start:stop], flipped)
        for row, image_index in enumerate(range(start, stop)):
            distance[row, image_index] = 64
            partners = np.flatnonzero(distance[row, :image_index] <= IDENTICAL_HASH_MAX_DISTANCE)
            for partner in partners:
                union.union(image_index, int(partner))
                identical_edges += 1
            mirror_partners = np.flatnonzero(mirror_distance[row] <= IDENTICAL_HASH_MAX_DISTANCE)
            for partner in mirror_partners:
                if int(partner) == image_index:
                    continue
                union.union(image_index, int(partner))
                if int(partner) < image_index:
                    mirror_edges += 1
        nearest[start:stop] = distance.min(axis=1)
        if stop == count or stop % 2048 == 0:
            print(f"Compared perceptual hashes: {stop}/{count}")

    bins = [0, 1, 6, 11, 16, 21, 31, 65]
    histogram = {}
    for low, high in zip(bins, bins[1:]):
        histogram[f"{low}-{high - 1}"] = int(((nearest >= low) & (nearest < high)).sum())
    stats = {
        "nearest_distance_percentiles": {
            "min": int(nearest.min()),
            "p50": int(np.percentile(nearest, 50)),
            "p90": int(np.percentile(nearest, 90)),
            "p99": int(np.percentile(nearest, 99)),
            "max": int(nearest.max()),
        },
        "nearest_distance_histogram": histogram,
        "images_with_identical_neighbor": int((nearest <= IDENTICAL_HASH_MAX_DISTANCE).sum()),
        "identical_hash_pair_count": identical_edges,
        "mirrored_hash_pair_count": mirror_edges,
    }
    return stats, identical_edges, mirror_edges


def build_groups(records: list[dict], hashes: np.ndarray) -> tuple[dict, dict]:
    count = len(records)
    union = UnionFind(count)
    reasons: Counter[str] = Counter()

    md5_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        md5_to_indices[record["md5"]].append(index)
    for indices in md5_to_indices.values():
        for other in indices[1:]:
            union.union(indices[0], other)
            reasons["exact_duplicate"] += 1

    print("Comparing perceptual hashes...")
    similarity, identical_edges, mirror_edges = compare_hashes(hashes, union)
    reasons["identical_hash"] = identical_edges
    reasons["mirrored_hash"] = mirror_edges

    sequence_buckets: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    video_buckets: dict[str, list[int]] = defaultdict(list)
    unmatched_stems: set[str] = set()
    for index, record in enumerate(records):
        video_name = video_prefix(record["source_stem"])
        if video_name:
            video_buckets[video_name].append(index)
        parsed = sequence_key(record["source_stem"])
        if parsed is None:
            unmatched_stems.add(record["source_stem"])
            continue
        prefix, frame = parsed
        if video_name:
            continue
        sequence_buckets[prefix][frame].append(index)

    sequence_links = 0
    for frames in sequence_buckets.values():
        ordered_frames = sorted(frames)
        for previous_frame, current_frame in zip(ordered_frames, ordered_frames[1:]):
            if current_frame - previous_frame > SEQUENCE_MAX_FRAME_GAP:
                continue
            for left in frames[previous_frame]:
                for right in frames[current_frame]:
                    if original_distance(hashes, left, right) <= SEQUENCE_MAX_DISTANCE:
                        union.union(left, right)
                        sequence_links += 1
    reasons["sequence_neighbor"] = sequence_links

    for indices in video_buckets.values():
        for other in indices[1:]:
            union.union(indices[0], other)
            reasons["video_filename"] += 1

    grouped = union.groups()
    groups = []
    for group_index, member_indexes in enumerate(sorted(grouped.values(), key=lambda items: items[0])):
        class_boxes = np.zeros(len(CLASS_NAMES), dtype=np.int64)
        prefixes = Counter()
        for member in member_indexes:
            for class_id, *_rest in records[member]["boxes"]:
                class_boxes[class_id] += 1
            parsed = sequence_key(records[member]["source_stem"])
            if parsed:
                prefixes[parsed[0]] += 1
        groups.append(
            {
                "id": group_index,
                "members": member_indexes,
                "class_boxes": class_boxes,
                "prefixes": prefixes,
            }
        )

    multi = [group for group in groups if len(group["members"]) > 1]
    largest = sorted(groups, key=lambda group: len(group["members"]), reverse=True)[:15]
    indexes_by_stem: dict[str, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        indexes_by_stem[record["source_stem"]].append(index)
    shared_stem_distances = []
    stems_with_copies = 0
    visually_different_shared_stems = 0
    for indexes in indexes_by_stem.values():
        if len(indexes) < 2:
            continue
        stems_with_copies += 1
        distances = [
            original_distance(hashes, left, right)
            for left_pos, left in enumerate(indexes)
            for right in indexes[left_pos + 1 :]
        ]
        minimum = min(distances) if distances else 64
        shared_stem_distances.append(minimum)
        if minimum > IDENTICAL_HASH_MAX_DISTANCE:
            visually_different_shared_stems += 1

    return {
        "groups": groups,
        "link_counts": dict(reasons),
        "summary": {
            "group_count": len(groups),
            "multi_image_group_count": len(multi),
            "images_in_multi_image_groups": int(sum(len(group["members"]) for group in multi)),
            "largest_group_size": int(max(len(group["members"]) for group in groups)),
            "largest_groups": [
                {
                    "group_id": group["id"],
                    "image_count": len(group["members"]),
                    "prefixes": dict(group["prefixes"]),
                    "examples": [records[index]["filename"] for index in group["members"][:5]],
                }
                for group in largest
                if len(group["members"]) > 1
            ],
            "filename_patterns": {
                "sequence_prefixes": {prefix: int(sum(len(v) for v in frames.values())) for prefix, frames in sorted(sequence_buckets.items())},
                "unmatched_source_stems": sorted(unmatched_stems)[:30],
                "unmatched_source_stem_count": len(unmatched_stems),
            },
            "shared_filename_stems": {
                "unique_source_stems": len(indexes_by_stem),
                "stems_with_multiple_files": stems_with_copies,
                "stems_with_no_identical_pair": visually_different_shared_stems,
                "minimum_within_stem_distance_percentiles": _percentiles(shared_stem_distances),
            },
            "video_prefixes": {name: len(indexes) for name, indexes in sorted(video_buckets.items())},
        },
        "similarity": similarity,
    }


def _percentiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "max": None}
    array = np.array(values)
    return {
        "min": int(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "max": int(array.max()),
    }


def label_statistics(records: list[dict]) -> dict:
    class_boxes = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    class_images = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    objects_per_image = []
    widths = []
    heights = []
    areas = []
    issue_counts: Counter[str] = Counter()
    issue_examples: dict[str, list[str]] = defaultdict(list)
    outside = 0
    tiny = 0
    empty_labels = 0

    for record in records:
        for issue in record["label_issues"]:
            kind = issue.split(":", 1)[0]
            issue_counts[kind] += 1
            if len(issue_examples[kind]) < 20:
                issue_examples[kind].append(record["filename"])
        if any(issue == "empty_label" for issue in record["label_issues"]):
            empty_labels += 1
        present = set()
        objects_per_image.append(len(record["boxes"]))
        for class_id, cx, cy, width, height in record["boxes"]:
            class_boxes[class_id] += 1
            present.add(class_id)
            widths.append(width)
            heights.append(height)
            areas.append(width * height)
            x1, y1 = cx - width / 2, cy - height / 2
            x2, y2 = cx + width / 2, cy + height / 2
            if x1 < -1e-4 or y1 < -1e-4 or x2 > 1 + 1e-4 or y2 > 1 + 1e-4:
                outside += 1
            if width * height < TINY_BOX_AREA:
                tiny += 1
        for class_id in present:
            class_images[class_id] += 1

    object_counter = Counter(objects_per_image)
    return {
        "bounding_box_count": int(class_boxes.sum()),
        "empty_label_count": empty_labels,
        "class_box_counts": {name: int(class_boxes[index]) for index, name in enumerate(CLASS_NAMES)},
        "class_image_counts": {name: int(class_images[index]) for index, name in enumerate(CLASS_NAMES)},
        "objects_per_image": {
            "min": int(min(objects_per_image) if objects_per_image else 0),
            "mean": float(np.mean(objects_per_image) if objects_per_image else 0),
            "median": float(np.median(objects_per_image) if objects_per_image else 0),
            "max": int(max(objects_per_image) if objects_per_image else 0),
            "histogram_head": {str(key): object_counter[key] for key in sorted(object_counter)[:21]},
        },
        "bbox_width_percentiles": _float_percentiles(widths),
        "bbox_height_percentiles": _float_percentiles(heights),
        "bbox_area_percentiles": _float_percentiles(areas),
        "boxes_extending_outside_image": outside,
        "tiny_boxes": tiny,
        "label_issue_counts": dict(issue_counts),
        "label_issue_examples": dict(issue_examples),
    }


def _float_percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "p50": None, "p90": None, "p99": None, "max": None}
    array = np.array(values, dtype=np.float64)
    return {
        "min": float(array.min()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def assign_splits(groups: list[dict]) -> dict[int, str]:
    rng = np.random.default_rng(RANDOM_SEED)
    total_images = sum(len(group["members"]) for group in groups)
    class_totals = sum((group["class_boxes"] for group in groups), start=np.zeros(len(CLASS_NAMES), dtype=np.int64))
    target_images = {name: ratio * total_images for name, ratio in SPLIT_RATIOS.items()}
    target_classes = {name: ratio * class_totals for name, ratio in SPLIT_RATIOS.items()}
    image_counts = {name: 0 for name in SPLIT_RATIOS}
    class_counts = {name: np.zeros(len(CLASS_NAMES), dtype=np.int64) for name in SPLIT_RATIOS}

    tie_break = rng.random(len(groups))
    order = sorted(range(len(groups)), key=lambda index: (-len(groups[index]["members"]), tie_break[index]))
    assignment: dict[int, str] = {}
    val_limit = target_images["val"]

    for index in order:
        group = groups[index]
        size = len(group["members"])
        eligible = ["train"] if size > val_limit * 1.15 else ["train", "val", "test"]
        best_split = None
        best_score = -1e18
        for split in eligible:
            if split != "train" and image_counts[split] + size > target_images[split]:
                continue
            image_deficit = target_images[split] - image_counts[split]
            weights = group["class_boxes"]
            if weights.sum() > 0:
                class_need_vector = (target_classes[split] - class_counts[split]) / np.maximum(target_classes[split], 1)
                class_need = float(np.dot(weights, class_need_vector) / weights.sum())
            else:
                class_need = 0.0
            score = image_deficit + 0.05 * total_images * class_need
            if score > best_score:
                best_score = score
                best_split = split
        if best_split is None:
            best_split = "train"
        assignment[group["id"]] = best_split
        image_counts[best_split] += size
        class_counts[best_split] += group["class_boxes"]
    return assignment


def write_split(records: list[dict], groups: list[dict], assignment: dict[int, str]) -> dict:
    if SPLIT_ROOT.exists():
        shutil.rmtree(SPLIT_ROOT)
    for split in SPLIT_RATIOS:
        (SPLIT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (SPLIT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    split_boxes = {split: np.zeros(len(CLASS_NAMES), dtype=np.int64) for split in SPLIT_RATIOS}
    split_images = {split: 0 for split in SPLIT_RATIOS}
    copied = 0
    for group in groups:
        split = assignment[group["id"]]
        for member in group["members"]:
            record = records[member]
            if record["label_path"] is None:
                raise RuntimeError(f"Missing label for {record['filename']}")
            destination_image = SPLIT_ROOT / "images" / split / record["filename"]
            destination_label = SPLIT_ROOT / "labels" / split / record["label_path"].name
            shutil.copy2(record["image_path"], destination_image)
            shutil.copy2(record["label_path"], destination_label)
            split_images[split] += 1
            for class_id, *_rest in record["boxes"]:
                split_boxes[split][class_id] += 1
            manifest_rows.append(
                {
                    "filename": record["filename"],
                    "split": split,
                    "group_id": group["id"],
                    "group_size": len(group["members"]),
                    "source_stem": record["source_stem"],
                }
            )
            copied += 1
            if copied % 1000 == 0 or copied == len(records):
                print(f"Copied split files: {copied}/{len(records)}")

    yaml_text = "\n".join(
        [
            "# TrafficMind AI grouped split. Generated by scripts/prepare_dataset.py.",
            f"# random_seed: {RANDOM_SEED}",
            "# ratios: train 0.80, val 0.10, test 0.10",
            "# Groups keep exact duplicates, identical dHashes, similar neighboring",
            "# frames (dHash <= 12, frame gap <= 3), and explicit *_mp4 frames together.",
            "train: images/train",
            "val: images/val",
            "test: images/test",
            "",
            f"nc: {len(CLASS_NAMES)}",
            "names:",
            *[f"  {index}: {name}" for index, name in enumerate(CLASS_NAMES)],
            "",
        ]
    )
    (SPLIT_ROOT / "data.yaml").write_text(yaml_text, encoding="utf-8")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = REPORT_DIR / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["filename", "split", "group_id", "group_size", "source_stem"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return {
        "image_counts": split_images,
        "class_box_counts": {
            split: {name: int(split_boxes[split][index]) for index, name in enumerate(CLASS_NAMES)}
            for split in SPLIT_RATIOS
        },
        "manifest": str(manifest_path.relative_to(PROJECT_ROOT)),
    }


def verify_split(records: list[dict], groups: list[dict], assignment: dict[int, str], split_summary: dict) -> None:
    original_images = len(list(IMAGE_DIR.glob("*.jpg")))
    original_labels = len(list(LABEL_DIR.glob("*.txt")))
    if original_images != len(records) or original_labels != len(records):
        raise RuntimeError("Original dataset counts changed while preparing the split.")

    seen_groups: dict[int, str] = {}
    copied_names = []
    for split in SPLIT_RATIOS:
        image_names = {path.name for path in (SPLIT_ROOT / "images" / split).glob("*.jpg")}
        label_names = {path.stem for path in (SPLIT_ROOT / "labels" / split).glob("*.txt")}
        if {Path(name).stem for name in image_names} != label_names:
            raise RuntimeError(f"Image/label mismatch in split {split}")
        if len(image_names) != split_summary["image_counts"][split]:
            raise RuntimeError(f"Unexpected image count in split {split}")
        copied_names.extend(image_names)
    if sorted(copied_names) != sorted(record["filename"] for record in records):
        raise RuntimeError("Split filenames do not match the original dataset.")
    for group in groups:
        split = assignment[group["id"]]
        if group["id"] in seen_groups and seen_groups[group["id"]] != split:
            raise RuntimeError(f"Group {group['id']} was assigned twice.")
        seen_groups[group["id"]] = split

    yaml_lines = (SPLIT_ROOT / "data.yaml").read_text(encoding="utf-8")
    for name in CLASS_NAMES:
        if name not in yaml_lines:
            raise RuntimeError(f"Class {name} is missing from data.yaml")
    for relative in ("images/train", "images/val", "images/test", "labels/train", "labels/val", "labels/test"):
        if not (SPLIT_ROOT / relative).is_dir():
            raise RuntimeError(f"Missing split directory: {relative}")


def clean_cache() -> None:
    for path in (CACHE_HASHES, CACHE_META):
        if path.exists():
            path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit and split the TrafficMind dataset.")
    parser.add_argument("--audit-only", action="store_true", help="Audit and group without copying the split.")
    args = parser.parse_args()

    print("Loading image and label inventory...")
    records, inventory = load_records()
    print("Scanning image bytes, dimensions, and perceptual hashes...")
    hashes, scan_summary = scan_images(records)
    print("Building leakage groups...")
    grouping = build_groups(records, hashes)
    statistics = label_statistics(records)

    report = {
        "inventory": inventory,
        "scan": scan_summary,
        "labels": statistics,
        "similarity": grouping["similarity"],
        "grouping": grouping["summary"] | {"link_counts": grouping["link_counts"]},
        "parameters": {
            "random_seed": RANDOM_SEED,
            "ratios": SPLIT_RATIOS,
            "identical_hash_max_distance": IDENTICAL_HASH_MAX_DISTANCE,
            "sequence_max_distance": SEQUENCE_MAX_DISTANCE,
            "sequence_max_frame_gap": SEQUENCE_MAX_FRAME_GAP,
            "class_names": CLASS_NAMES,
        },
    }

    if not args.audit_only:
        print("Assigning grouped train/validation/test split...")
        assignment = assign_splits(grouping["groups"])
        split_summary = write_split(records, grouping["groups"], assignment)
        verify_split(records, grouping["groups"], assignment, split_summary)
        total = sum(split_summary["image_counts"].values())
        report["split"] = {
            "image_counts": split_summary["image_counts"],
            "image_ratios": {split: count / total for split, count in split_summary["image_counts"].items()},
            "class_box_counts": split_summary["class_box_counts"],
            "output_dir": str(SPLIT_ROOT.relative_to(PROJECT_ROOT)),
            "data_yaml": "dataset_split/data.yaml",
            "manifest": split_summary["manifest"],
        }
        clean_cache()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / "dataset_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
