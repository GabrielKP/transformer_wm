"""Utility functions for data reading.
"""

import json
import os
from hashlib import shake_128
from typing import Dict, List, Union

from lm_mem import get_logger

logger = get_logger(__name__)


def compute_hash(text: str) -> str:
    h = shake_128(text.encode("utf-8"))
    return h.hexdigest(5)


def read_vignettes(path: str) -> List[Dict]:
    with open(path) as f_in:
        vignettes = json.load(f_in)

    return vignettes["vignettes"]


def save_multiple_to_json(
    outputs: List[Union[Dict[str, List[Dict]], List[List[Dict]]]],
    output_paths: List[str],
) -> None:
    for output, output_path in zip(outputs, output_paths):
        if isinstance(output, list):
            save_to_json(
                {"sequences": output},
                output_path,
            )
        else:
            save_to_json(output, output_path)


def save_to_json(
    output_dict: List[Dict],
    output_path: str,
) -> None:
    # Save to output_file
    with open(output_path, "w") as f_out:
        json.dump(output_dict, f_out)
        f_out.write("\n")


def read_txt(
    file_path: str,
) -> List[str]:
    """Reads a txt file, can handle comments with #."""
    lines = []
    with open(file_path, "r", encoding="utf-8") as f_in:
        for line in f_in.readlines():
            # Handle comment lines
            if line[0] == "#":
                continue
            # Remove newline
            if line.endswith("\n"):
                line = line[:-1]
            # Handle comments and trailing whitespaces until comments
            space_counter = 0
            for idx, char in enumerate(line):
                if char == " ":
                    space_counter += 1
                else:
                    space_counter = 0
                if char == "#":
                    line = line[: idx - space_counter]
                    break
            # add line
            lines.append(line)
    return lines


def save_with_suffix(
    output_dir: str,
    seq1s: List[Dict],
    seq2s: List[Dict],
    suffix: str,
) -> None:
    seq1s_path = os.path.join(output_dir, f"seq1s_{suffix}.json")
    logger.info(f"{suffix}: Writing seq1s to {seq1s_path}")
    save_to_json(
        {"sequences": seq1s},
        seq1s_path,
    )
    seq2s_path = os.path.join(output_dir, f"seq2s_{suffix}.json")
    logger.info(f"{suffix}: Writing seq2s to {seq2s_path}")
    save_to_json(
        {"sequences": seq2s},
        seq2s_path,
    )


def print_synonyms(path: str) -> None:
    """Prints synyonyms such that they can be copied into a table

    Lines starting with '#' are ignored.
    Lines starting with '--' indicate original word. Count as markers to start new list.
    Lines below indicate synonyms for that word.
    """
    synonyms: List[List[str]] = []
    current_word = None
    with open(path, "r") as f_in:
        for line in f_in.readlines():
            if line.startswith("#"):
                continue
            if line.endswith("\n"):
                line = line[:-1]
            if line.startswith("--"):
                current_word = line.replace("--", "").strip()
                synonyms.append([current_word])
            elif current_word is None:
                raise ValueError("Missing base word in synonyms file.")
            else:
                synonyms[-1].append(line.strip())

    print("\n".join([", ".join(group) for group in synonyms]))
