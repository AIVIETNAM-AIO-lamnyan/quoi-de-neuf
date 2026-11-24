#!/usr/bin/env python3
"""
Text Preprocessing Script for Vietnamese Social Media Data.

This script preprocesses CSV data by:
1. Detecting and excluding non-Vietnamese posts (foreign languages, pure emojis/punctuation, empty posts).
2. Removing all icons, emojis, emoticons, pictographs, and symbol artifacts.
3. Converting Vietnamese teencode / slang into standard Vietnamese words using teencode.txt.
4. Preserving URLs, domains, emails, mentions, hashtags, and token capitalization.

Usage:
    python preprocess/preprocess.py
    python preprocess/preprocess.py --input data/combined.csv --output data/combined_preprocessed.csv
    python preprocess/preprocess.py --keep-non-vn  # Keep non-Vietnamese rows without filtering
    python preprocess/preprocess.py --inplace      # Overwrite input file directly
"""

import argparse
import csv
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

# Comprehensive Unicode Emoji / Icon Pattern
ICON_PATTERN = re.compile(
    r'('
    r'[\U0001F600-\U0001F64F]'  # Emoticons
    r'|[\U0001F300-\U0001F5FF]'  # Misc Symbols and Pictographs
    r'|[\U0001F680-\U0001F6FF]'  # Transport and Map
    r'|[\U0001F700-\U0001F77F]'  # Alchemical Symbols
    r'|[\U0001F780-\U0001F7FF]'  # Geometric Shapes Extended
    r'|[\U0001F800-\U0001F8FF]'  # Supplemental Arrows-C
    r'|[\U0001F900-\U0001F9FF]'  # Supplemental Symbols and Pictographs
    r'|[\U0001FA00-\U0001FA6F]'  # Chess Symbols / Extended-A
    r'|[\U0001FA70-\U0001FAFF]'  # Symbols and Pictographs Extended-A
    r'|[\U0001FB00-\U0001FBFF]'  # Symbols Extended-B
    r'|[\U0001F1E6-\U0001F1FF]'  # Regional indicator symbols (country flags)
    r'|[\U0001F3FB-\U0001F3FF]'  # Emoji modifier skin tones
    r'|[\u2600-\u26FF]'          # Misc symbols (e.g. ☕, ⚠, ⚡)
    r'|[\u2700-\u27BF]'          # Dingbats (e.g. ✨, ❄, ❣)
    r'|[\u2300-\u23FF]'          # Misc Technical (e.g. ⏰)
    r'|[\u2B00-\u2BFF]'          # Misc Symbols and Arrows (e.g. ⬛, ⭐)
    r'|[\u2580-\u259F]'          # Block elements (e.g. ▓)
    r'|[\u25A0-\u25FF]'          # Geometric shapes
    r'|[\u2190-\u21FF]'          # Arrows
    r'|[\u203C\u2049\u2122\u2139\u00A9\u00AE]'  # Special symbols
    r'|[\u200D\uFE0E\uFE0F]'     # ZWJ and variation selectors
    r'|[0-9#*]\uFE0F?\u20E3'     # Enclosing Keycaps (e.g. 1️⃣)
    r')+',
    flags=re.UNICODE
)

# Pattern to protect structured entities (URLs, domains, emails, handles, hashtags)
PROTECT_PATTERN = (
    r'https?://\S+'
    r'|www\.\S+'
    r'|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?'
    r'|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
    r'|@[\w.-]+'
    r'|#[\w.-]+'
)

# Vietnamese diacritics matcher
VN_DIACRITICS_RE = re.compile(
    r'[àáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
    r'ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]'
)

# Foreign non-Latin scripts (CJK, Korean, Japanese, Cyrillic, Arabic, Thai)
NON_LATIN_RE = re.compile(
    r'[\u4e00-\u9fff\u3400-\u4dbf\uac00-\ud7af\u3040-\u30ff\u0400-\u04ff\u0600-\u06ff\u0e00-\u0e7f]'
)

# Common high-frequency English stopwords for language disambiguation
COMMON_EN_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 'not', 'on', 'with',
    'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her',
    'she', 'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their', 'what', 'so', 'up',
    'out', 'if', 'about', 'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time',
    'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good', 'some', 'could',
    'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only', 'come', 'its', 'over', 'think',
    'also', 'back', 'after', 'use', 'two', 'how', 'our', 'work', 'first', 'well', 'way', 'even',
    'new', 'want', 'because', 'any', 'these', 'give', 'day', 'most', 'us', 'is', 'are', 'was', 'were'
}

# Common Vietnamese unaccented words and teencode/slang for accentless text detection
COMMON_VN_UNACCENTED = {
    'ko', 'k', 'khong', 'dc', 'đc', 'nguoi', 'ng', 'minh', 'toi', 'tui', 'tuii', 'ban', 'nay',
    'do', 'o', 'la', 'va', 'co', 'muon', 'thoi', 'nhe', 'nha', 'roi', 'chua', 'biet', 'thay',
    'nhieu', 'it', 'dep', 'xau', 'dung', 'sai', 'lam', 'gi', 'sao', 'nao', 'the', 'di', 've',
    'em', 'anh', 'chi', 'bac', 'ong', 'ba', 'tao', 'to', 'may', 'no', 'ho', 'chung', 'ai',
    'troi', 'oi', 'dcm', 'vcl', 'bcuoi', 'vl', 'clb', 'nckh', 'khum', 'hong', 'hongg', 'nx',
    'chx', 'trg', 'hc', 'cs', 'kbt', 'dhs', 'khs', 'v~', 'z', 'zậy', 'zui', 'ze', 'hok'
}


def is_vietnamese(text: Optional[str]) -> bool:
    """
    Determine whether text is Vietnamese using a general heuristic:
    1. Rejects empty, whitespace-only, or pure link/mention texts.
    2. Recognizes texts containing Vietnamese diacritics (unless foreign non-Latin scripts dominate).
    3. For unaccented texts, checks vocabulary overlap between English and Vietnamese slang/stopwords.
    """
    if not text:
        return False

    # Remove URLs, mentions, hashtags
    clean = re.sub(r'https?://\S+|www\.\S+|@\S+|#\S+', ' ', text)

    # Extract words
    words = re.findall(
        r'[a-zA-Zàáảãạâầấẩẫậăằắẳẵặèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ'
        r'ÀÁẢÃẠÂẦẤẨẪẬĂẰẮẲẴẶÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴĐ]+',
        clean
    )
    if not words:
        return False

    has_vn_diacritics = bool(VN_DIACRITICS_RE.search(clean))
    non_latin_count = len(NON_LATIN_RE.findall(clean))

    # 1. Text with Vietnamese diacritics
    if has_vn_diacritics:
        vn_diacritic_count = len(VN_DIACRITICS_RE.findall(clean))
        # Drop if dominated by foreign non-Latin characters (Chinese, Japanese, etc.)
        if non_latin_count > vn_diacritic_count * 2:
            return False
        return True

    # 2. Text without Vietnamese diacritics
    if non_latin_count > 0:
        return False

    words_lower = [w.lower() for w in words]
    en_matches = sum(1 for w in words_lower if w in COMMON_EN_WORDS)
    vn_matches = sum(1 for w in words_lower if w in COMMON_VN_UNACCENTED)

    if vn_matches > en_matches:
        return True
    if en_matches > 0:
        return False

    return vn_matches > 0


def remove_icons(text: Optional[str]) -> str:
    """
    Remove all icons, emojis, pictographs, and related variation selectors from text.
    Also normalizes horizontal whitespace on each line, preserving line breaks.
    """
    if not text:
        return "" if text is not None else None

    # Strip icon sequences
    cleaned = ICON_PATTERN.sub("", text)

    # Clean leftover zero-width characters and BOMs
    cleaned = re.sub(r"[\u200B-\u200D\uFEFF]", "", cleaned)

    # Normalize horizontal whitespace per line
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in cleaned.split("\n")]
    return "\n".join(lines)


def load_teencode_dict(dict_path: str) -> Dict[str, str]:
    """
    Load teencode dictionary from a tab-separated file.
    Format: <teencode>\t<standard_word>
    """
    teencode_dict = {}
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Teencode dictionary not found at: {dict_path}")

    with open(dict_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                key = parts[0].strip().lower()
                value = parts[1].strip()
                teencode_dict[key] = value

    return teencode_dict


def build_teencode_pattern(teencode_dict: Dict[str, str]) -> re.Pattern:
    """
    Build an optimized regular expression pattern matching teencode entries.
    Longer multi-word phrases are prioritized first to avoid partial replacements.
    Protected entities (URLs, emails, handles) are matched first to prevent mangling.
    """
    sorted_keys = sorted(
        teencode_dict.keys(),
        key=lambda x: (len(x.split()), len(x)),
        reverse=True
    )

    teencode_patterns = []
    for k in sorted_keys:
        words = [re.escape(w) for w in k.split()]
        teencode_patterns.append(r"\s+".join(words))

    full_pattern_str = (
        rf"(?P<protect>{PROTECT_PATTERN})"
        rf"|(?P<teencode>(?<![-\w])(?:{'|'.join(teencode_patterns)})(?![-\w]))"
    )

    return re.compile(full_pattern_str, flags=re.IGNORECASE)


def match_case(orig: str, replacement: str) -> str:
    """
    Match the capitalization style of the original token:
    - UPPERCASE: returns replacement in UPPERCASE
    - Capitalized / Titlecase: returns replacement with first letter capitalized
    - lowercase / other: returns replacement as is
    """
    if orig.isupper():
        return replacement.upper()
    elif orig and orig[0].isupper():
        return replacement[0].upper() + replacement[1:]
    return replacement


def convert_teencode(
    text: Optional[str],
    teencode_dict: Dict[str, str],
    pattern: re.Pattern
) -> str:
    """
    Convert Vietnamese teencode words to standard words using the compiled pattern
    and teencode dictionary while preserving protected entities and casing.
    """
    if not text:
        return "" if text is not None else None

    def _replace_match(match: re.Match) -> str:
        if match.group("protect"):
            return match.group("protect")

        matched_str = match.group("teencode")
        if not matched_str:
            return match.group(0)

        normalized_key = re.sub(r"\s+", " ", matched_str.lower().strip())
        standard_word = teencode_dict.get(normalized_key, matched_str)
        return match_case(matched_str, standard_word)

    return pattern.sub(_replace_match, text)


def preprocess_text(
    text: Optional[str],
    teencode_dict: Dict[str, str],
    pattern: re.Pattern
) -> str:
    """
    Execute full text preprocessing pipeline:
    1. Remove any icon/emoji.
    2. Convert Vietnamese teencode to standard Vietnamese words.
    """
    if text is None:
        return ""

    no_icons = remove_icons(text)
    return convert_teencode(no_icons, teencode_dict, pattern)


def resolve_path(path_str: str, base_dirs: List[str], is_output: bool = False) -> str:
    """Resolve a relative or absolute file path against potential base directories."""
    if os.path.isabs(path_str):
        return path_str

    if is_output:
        for base in base_dirs:
            candidate = os.path.abspath(os.path.join(base, path_str))
            if os.path.exists(os.path.dirname(candidate)):
                return candidate
        return os.path.abspath(os.path.join(base_dirs[0], path_str))

    for base in base_dirs:
        candidate = os.path.abspath(os.path.join(base, path_str))
        if os.path.exists(candidate):
            return candidate

    return os.path.abspath(os.path.join(base_dirs[0], path_str))


def preprocess_csv(
    input_file: str,
    output_file: str,
    teencode_file: str,
    text_column: str = "text",
    filter_non_vn: bool = True
) -> Tuple[int, int, int, int]:
    """
    Preprocess CSV file:
    - Excludes non-Vietnamese posts if filter_non_vn is True.
    - Removes icons / emojis.
    - Converts teencode to standard words in the specified text column.
    """
    print(f"Reading input file:      {input_file}")
    print(f"Teencode dictionary:     {teencode_file}")
    print(f"Target text column:      '{text_column}'")
    print(f"Filter non-Vietnamese:   {filter_non_vn}")

    teencode_dict = load_teencode_dict(teencode_file)
    print(f"Loaded {len(teencode_dict)} teencode dictionary mapping(s).")

    teencode_pattern = build_teencode_pattern(teencode_dict)

    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found at: {input_file}")

    total_rows = 0
    excluded_non_vn_count = 0
    kept_rows = 0
    icons_removed_count = 0
    teencode_converted_count = 0

    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    temp_output_file = output_file + ".tmp" if os.path.abspath(input_file) == os.path.abspath(output_file) else output_file

    start_time = time.time()

    with open(input_file, "r", newline="", encoding="utf-8-sig", errors="replace") as in_f:
        reader = csv.DictReader(in_f)
        fieldnames = reader.fieldnames

        if not fieldnames or text_column not in fieldnames:
            raise ValueError(
                f"Column '{text_column}' not found in CSV. Available columns: {fieldnames}"
            )

        with open(temp_output_file, "w", newline="", encoding="utf-8-sig") as out_f:
            writer = csv.DictWriter(out_f, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                total_rows += 1
                orig_text = row.get(text_column) or ""

                # Step 1: Check language filter
                if filter_non_vn and not is_vietnamese(orig_text):
                    excluded_non_vn_count += 1
                    continue

                kept_rows += 1

                # Step 2: Remove icons
                no_icons = remove_icons(orig_text)
                if no_icons != orig_text:
                    icons_removed_count += 1

                # Step 3: Convert teencode
                processed_text = convert_teencode(no_icons, teencode_dict, teencode_pattern)
                if processed_text != no_icons:
                    teencode_converted_count += 1

                row[text_column] = processed_text
                writer.writerow(row)

    if temp_output_file != output_file:
        os.replace(temp_output_file, output_file)

    elapsed = time.time() - start_time
    print(f"\nProcessing Summary:")
    print(f"  - Total rows read:              {total_rows:,}")
    if filter_non_vn:
        print(f"  - Non-Vietnamese rows excluded: {excluded_non_vn_count:,}")
        print(f"  - Vietnamese rows kept:         {kept_rows:,}")
    print(f"  - Rows with icons removed:      {icons_removed_count:,}")
    print(f"  - Rows with teencode converted: {teencode_converted_count:,}")
    print(f"  - Time elapsed:                 {elapsed:.2f} seconds")
    print(f"  - Output saved to:              {os.path.abspath(output_file)}")

    return total_rows, excluded_non_vn_count, icons_removed_count, teencode_converted_count


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    search_dirs = [os.getcwd(), project_root, script_dir]

    parser = argparse.ArgumentParser(
        description="Preprocess text in CSV: exclude non-Vietnamese posts, remove icons, and convert teencode."
    )
    parser.add_argument(
        "-i", "--input",
        default="data/combined.csv",
        help="Path to input CSV file (default: data/combined.csv)."
    )
    parser.add_argument(
        "-o", "--output",
        default="data/combined_preprocessed.csv",
        help="Path to output CSV file (default: data/combined_preprocessed.csv)."
    )
    parser.add_argument(
        "-t", "--teencode",
        default="preprocess/teencode.txt",
        help="Path to teencode dictionary file (default: preprocess/teencode.txt)."
    )
    parser.add_argument(
        "-c", "--column",
        default="text",
        help="Name of the text column to preprocess (default: 'text')."
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input CSV file directly."
    )
    parser.add_argument(
        "--keep-non-vn",
        action="store_true",
        help="Keep non-Vietnamese posts instead of excluding them."
    )

    args = parser.parse_args()

    input_path = resolve_path(args.input, search_dirs)
    teencode_path = resolve_path(args.teencode, search_dirs)

    if args.inplace:
        output_path = input_path
    else:
        output_path = resolve_path(args.output, search_dirs, is_output=True)

    preprocess_csv(
        input_file=input_path,
        output_file=output_path,
        teencode_file=teencode_path,
        text_column=args.column,
        filter_non_vn=not args.keep_non_vn
    )


if __name__ == "__main__":
    main()
