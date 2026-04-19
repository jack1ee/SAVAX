#!/usr/bin/env python3
"""Convert official EgoMe annotations into the paired JSON/paragraph format used by this repo."""

import argparse
import json
from pathlib import Path


DEFAULT_SPLITS = ("train", "val", "test", "total")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert official EgoMe annotation files (train/val/test/total.json) "
            "into paired *_Ego_cap.json / *_Exo_cap.json and *_Ego_para.json / *_Exo_para.json files."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the official EgoMe annotation JSON files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write the converted cap/para JSON files.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(DEFAULT_SPLITS),
        help="Split names to convert. Default: train val test total",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize the conversion without writing output files.",
    )
    return parser.parse_args()


def normalize_video_key(video_name):
    if not isinstance(video_name, str) or not video_name:
        raise ValueError("video name must be a non-empty string")
    return Path(video_name).stem


def strip_view_suffix(video_key):
    for suffix in ("_ego", "_exo"):
        if video_key.endswith(suffix):
            return video_key[: -len(suffix)]
    return video_key


def build_repo_entry(raw_key, raw_entry):
    fine_level = raw_entry.get("Fine-level", [])
    if not fine_level:
        raise ValueError(f"{raw_key} has no Fine-level annotations")

    ordered_steps = sorted(fine_level, key=lambda step: step.get("Step order", 0))
    timestamps = []
    sentences = []
    skipped_empty_steps = 0

    for index, step in enumerate(ordered_steps):
        if "Step timestamp" not in step or "Step discription" not in step:
            raise KeyError(f"{raw_key} step #{index} is missing timestamp or description")
        timestamp = step["Step timestamp"]
        if not isinstance(timestamp, (list, tuple)) or len(timestamp) != 2:
            raise ValueError(f"{raw_key} step #{index} has invalid timestamp {timestamp}")

        start = float(timestamp[0])
        end = float(timestamp[1])
        if start > end:
            raise ValueError(f"{raw_key} step #{index} has reversed timestamp {timestamp}")

        sentence = str(step["Step discription"]).strip()
        if not sentence:
            skipped_empty_steps += 1
            continue

        timestamps.append([start, end])
        sentences.append(sentence)

    if not sentences:
        raise ValueError(f"{raw_key} has no usable Fine-level descriptions after filtering empty steps")

    duration = float(raw_entry["Duration"])
    if duration <= 0:
        raise ValueError(f"{raw_key} has non-positive duration {duration}")

    normalized_key = normalize_video_key(raw_key)
    match_video = raw_entry.get("Match video")
    normalized_match = normalize_video_key(match_video) if match_video else None

    return normalized_key, {
        "duration": duration,
        "timestamps": timestamps,
        "sentences": sentences,
        "coarse_sentence": raw_entry.get("Coarse-level", ""),
        "view": raw_entry.get("View", ""),
        "match_video": normalized_match,
        "video_type": raw_entry.get("Video type"),
        "scene": raw_entry.get("Scene", ""),
        "action": raw_entry.get("Action", ""),
        "split": raw_entry.get("Split", ""),
        "source_video": raw_key,
    }, skipped_empty_steps


def validate_pairs(ego_entries, exo_entries, split_name):
    ego_ids = {strip_view_suffix(key) for key in ego_entries}
    exo_ids = {strip_view_suffix(key) for key in exo_entries}
    missing_ego = sorted(exo_ids - ego_ids)
    missing_exo = sorted(ego_ids - exo_ids)
    if missing_ego or missing_exo:
        raise ValueError(
            f"{split_name}: unmatched ego/exo pairs detected. "
            f"missing_ego={len(missing_ego)} missing_exo={len(missing_exo)}"
        )


def convert_split(input_path):
    with input_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if "annotations" not in data or not isinstance(data["annotations"], dict):
        raise KeyError(f"{input_path} must contain a top-level 'annotations' dictionary")

    ego_entries = {}
    exo_entries = {}
    skipped_empty_steps = 0
    skipped_videos = []

    for raw_key, raw_entry in data["annotations"].items():
        if "View" not in raw_entry or "Duration" not in raw_entry:
            raise KeyError(f"{raw_key} is missing required official EgoMe fields")

        normalized_key, repo_entry, skipped_count = build_repo_entry(raw_key, raw_entry)
        if skipped_count:
            skipped_empty_steps += skipped_count
            skipped_videos.append(raw_key)
        view = str(raw_entry["View"]).strip().lower()

        if view == "ego":
            ego_entries[normalized_key] = repo_entry
        elif view == "exo":
            exo_entries[normalized_key] = repo_entry
        else:
            raise ValueError(f"{raw_key} has unsupported view '{raw_entry['View']}'")

    validate_pairs(ego_entries, exo_entries, input_path.stem)
    summary = {
        "skipped_empty_steps": skipped_empty_steps,
        "skipped_videos": skipped_videos,
    }
    return ego_entries, exo_entries, summary


def dump_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def build_para_entries(entries):
    para_entries = {}
    for video_key, info in entries.items():
        para_entries[video_key] = " ".join(
            sentence.strip() for sentence in info["sentences"] if sentence.strip()
        )
    return para_entries


def main():
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"input directory does not exist: {input_dir}")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in args.splits:
        input_path = input_dir / f"{split_name}.json"
        if not input_path.exists():
            raise FileNotFoundError(f"missing split file: {input_path}")

        ego_entries, exo_entries, summary = convert_split(input_path)
        pair_count = len(ego_entries)
        print(
            f"{split_name}: {pair_count} paired samples "
            f"({len(ego_entries)} ego / {len(exo_entries)} exo)"
        )
        if summary["skipped_empty_steps"]:
            print(
                f"  skipped {summary['skipped_empty_steps']} empty step descriptions "
                f"across {len(summary['skipped_videos'])} videos"
            )

        if args.dry_run:
            continue

        ego_cap_output = output_dir / f"{split_name}_Ego_cap.json"
        exo_cap_output = output_dir / f"{split_name}_Exo_cap.json"
        ego_para_output = output_dir / f"{split_name}_Ego_para.json"
        exo_para_output = output_dir / f"{split_name}_Exo_para.json"

        dump_json(ego_cap_output, ego_entries)
        dump_json(exo_cap_output, exo_entries)
        dump_json(ego_para_output, build_para_entries(ego_entries))
        dump_json(exo_para_output, build_para_entries(exo_entries))

        print(f"  wrote {ego_cap_output}")
        print(f"  wrote {exo_cap_output}")
        print(f"  wrote {ego_para_output}")
        print(f"  wrote {exo_para_output}")


if __name__ == "__main__":
    main()
