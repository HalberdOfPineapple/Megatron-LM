#!/usr/bin/env python

# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

"""Export a Hugging Face dataset saved with load_from_disk() to JSONL.

This is useful for turning Arrow datasets into the {"text": "..."} format
expected by tools/preprocess_data.py.
"""

import argparse
import json
from pathlib import Path

from datasets import Dataset, DatasetDict, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face dataset on disk to JSONL."
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=True,
        help="Path to a dataset directory created by datasets.load_from_disk().",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to the JSONL file to write.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default=None,
        help="Dataset split to export when the input path contains a DatasetDict.",
    )
    parser.add_argument(
        "--json-key",
        type=str,
        default="text",
        help="Field name to write in the output JSONL.",
    )
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Column to read from the input dataset.",
    )
    parser.add_argument(
        "--drop-empty",
        action="store_true",
        help="Skip rows where the selected text column is empty.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of rows to export.",
    )
    return parser.parse_args()


def get_dataset(dataset_or_dict, split):
    if isinstance(dataset_or_dict, Dataset):
        return dataset_or_dict

    if not isinstance(dataset_or_dict, DatasetDict):
        raise TypeError(f"Unsupported dataset type: {type(dataset_or_dict)}")

    if split is None:
        if "train" in dataset_or_dict:
            split = "train"
        else:
            split = next(iter(dataset_or_dict.keys()))

    if split not in dataset_or_dict:
        available = ", ".join(dataset_or_dict.keys())
        raise ValueError(f"Split '{split}' not found. Available splits: {available}")

    return dataset_or_dict[split]


def main():
    args = parse_args()

    dataset_or_dict = load_from_disk(args.input_path)
    dataset = get_dataset(dataset_or_dict, args.split)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as fout:
        for idx, example in enumerate(dataset):
            if args.limit is not None and written >= args.limit:
                break

            if idx >= 10: break
            print(f"Idx: {idx}, Example: {example.keys()}")
            continue

            if args.text_column not in example:
                available = ", ".join(example.keys())
                raise KeyError(
                    f"Column '{args.text_column}' not found in row {idx}. "
                    f"Available columns: {available}"
                )

            value = example[args.text_column]
            if value is None:
                text = ""
            elif isinstance(value, str):
                text = value
            else:
                text = json.dumps(value, ensure_ascii=False)

            if args.drop_empty and not text.strip():
                continue

            fout.write(json.dumps({args.json_key: text}, ensure_ascii=False) + "\n")
            written += 1

    print(f"Exported {written} rows to {output_path}")


if __name__ == "__main__":
    main()
