import json
import re
import argparse
from itertools import chain
from pathlib import Path

LONG_PARAGRAPH_THRESHOLD = 300
MIN_STEP_LENGTH = 20

# ---------------------------------------------------------------------------
# Phase 0: LaTeX placeholder protection
# ---------------------------------------------------------------------------

RE_DISPLAY_BRACKET = re.compile(r'\\\[.*?\\\]', re.DOTALL)
RE_DISPLAY_DOLLAR = re.compile(r'\$\$.*?\$\$', re.DOTALL)
RE_INLINE_PAREN = re.compile(r'\\\(.*?\\\)', re.DOTALL)
RE_INLINE_DOLLAR = re.compile(r'(?<!\$)\$(?!\$)(?!\s)(.+?)(?<!\s|\$)\$(?!\$)')


def protect_latex(text):
    """Replace all LaTeX regions with placeholders. Returns (protected_text, mapping)."""
    mapping = {}
    counter = [0]

    def _replace(m, kind):
        original = m.group(0)
        key = f"\x00{kind}_{counter[0]}\x00"
        mapping[key] = original
        counter[0] += 1
        return key

    # 1) Display math \[...\]
    text = RE_DISPLAY_BRACKET.sub(lambda m: _replace(m, "DISPLAY"), text)

    # 2) Display math $$...$$
    text = RE_DISPLAY_DOLLAR.sub(lambda m: _replace(m, "DISPLAY"), text)

    # 3) Inline math \(...\)
    text = RE_INLINE_PAREN.sub(lambda m: _replace(m, "INLINE"), text)

    # 4) Inline math $...$
    text = RE_INLINE_DOLLAR.sub(lambda m: _replace(m, "INLINE"), text)

    return text, mapping


def restore_latex(chunks, mapping):
    """Restore all LaTeX placeholders in a list of text chunks.

    Restores in reverse insertion order so that inner placeholders
    (captured earlier) are restored after outer ones that contain them.
    """
    # Reverse so last-inserted (outer) placeholders are restored first,
    # then their values (which may contain earlier placeholders) get
    # a second pass to resolve inner ones.
    keys_reversed = list(reversed(mapping))
    restored = []
    for chunk in chunks:
        # Multiple passes to handle nesting
        for _ in range(2):
            for key in keys_reversed:
                chunk = chunk.replace(key, mapping[key])
        restored.append(chunk)
    return restored


# ---------------------------------------------------------------------------
# Phase 1: Markdown header split
# ---------------------------------------------------------------------------

RE_MD_HEADER = re.compile(r'(?=^#{1,4}\s)', re.MULTILINE)


def split_markdown_headers(text):
    parts = RE_MD_HEADER.split(text)
    return [p for p in parts if p]


# ---------------------------------------------------------------------------
# Phase 2: Double newline split
# ---------------------------------------------------------------------------

RE_DOUBLE_NEWLINE = re.compile(r'\n\s*\n')


def split_double_newline(text):
    parts = RE_DOUBLE_NEWLINE.split(text)
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Phase 3: Display math block isolation
# ---------------------------------------------------------------------------

RE_DISPLAY_PLACEHOLDER = re.compile(r'(\x00DISPLAY_\d+\x00)')


def split_display_math(chunk):
    parts = RE_DISPLAY_PLACEHOLDER.split(chunk)
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Phase 4: Numbered list / bullet split
# ---------------------------------------------------------------------------

RE_NUMBERED_LIST = re.compile(r'(?=^\d+\.\s)', re.MULTILINE)
RE_BULLET = re.compile(r'(?=^- )', re.MULTILINE)


def split_numbered_and_bullets(chunk):
    # Try numbered list first
    parts = RE_NUMBERED_LIST.split(chunk)
    # Then split each part on bullets
    result = []
    for p in parts:
        sub = RE_BULLET.split(p)
        result.extend(s for s in sub if s.strip())
    return result if result else [chunk]


# ---------------------------------------------------------------------------
# Phase 5: Logical connector split (only for long chunks)
# ---------------------------------------------------------------------------

RE_LOGICAL_EN = re.compile(
    r'(?:(?<=\.\s)|(?<=\n)|(?<=:\n)|(?<=:\s))'
    r'(?=(?:Therefore|Thus|Hence|So,?\s|Now,?\s|Next,?\s|First,?\s|Then,?\s|'
    r'Since\s|Because\s|Note\sthat\s|'
    r'We\s(?:can|need|know|have|get|see|find|note|observe|use|compute|calculate)))'
)

RE_LOGICAL_ZH = re.compile(
    r'(?=(?:因此|所以|那么|接下来|首先|由于|已知))'
)


def split_logical_connectors(chunk, threshold=LONG_PARAGRAPH_THRESHOLD):
    if len(chunk) <= threshold:
        return [chunk]
    parts = RE_LOGICAL_EN.split(chunk)
    result = []
    for p in parts:
        if len(p) > threshold:
            sub = RE_LOGICAL_ZH.split(p)
            result.extend(s for s in sub if s.strip())
        elif p.strip():
            result.append(p)
    return result if result else [chunk]


# ---------------------------------------------------------------------------
# Phase 6: Sentence split for remaining long chunks
# ---------------------------------------------------------------------------

RE_SENTENCE = re.compile(r'(?<=\.)\s+(?=[A-Z])')


def split_sentences(chunk, threshold=LONG_PARAGRAPH_THRESHOLD):
    if len(chunk) <= threshold:
        return [chunk]
    parts = RE_SENTENCE.split(chunk)
    return [p for p in parts if p.strip()] or [chunk]


# ---------------------------------------------------------------------------
# Phase 7: Post-processing
# ---------------------------------------------------------------------------

# Regex to detect display math blocks (display math is the main content of the step)
RE_DISPLAY_MATH_ONLY = re.compile(
    r'^\s*(\\\[.*?\\\]|\$\$.*?\$\$)\s*$',
    re.DOTALL
)


def merge_display_math(steps):
    """Merge display math blocks with their preceding explanatory step.

    If step N is primarily a display math block (\\[...\\] or $$...$$),
    append it to step N-1 and remove step N. This reduces over-fragmentation
    caused by newlines separating formulas from their explanatory text.
    """
    if len(steps) <= 1:
        return steps

    merged = list(steps)
    i = 1
    while i < len(merged):
        step_text = merged[i]
        # Check if this step is primarily a display math block
        if RE_DISPLAY_MATH_ONLY.match(step_text):
            # Merge with previous step
            merged[i - 1] = merged[i - 1] + "\n" + step_text
            merged.pop(i)
            # Don't increment i - the current step shifted to the next position
        else:
            i += 1

    return merged


def postprocess(chunks, min_step=MIN_STEP_LENGTH):
    """Strip whitespace, remove empty chunks, merge too-short fragments."""
    cleaned = [c.strip() for c in chunks]
    cleaned = [c for c in cleaned if c]

    if not cleaned:
        return cleaned

    # Merge short fragments into the previous step
    merged = [cleaned[0]]
    for c in cleaned[1:]:
        if len(merged[-1]) < min_step:
            merged[-1] = merged[-1] + "\n" + c
        else:
            merged.append(c)
    # If the last step is too short, merge it back
    if len(merged) > 1 and len(merged[-1]) < min_step:
        merged[-2] = merged[-2] + "\n" + merged[-1]
        merged.pop()

    return merged


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def _flatten(list_of_lists):
    return list(chain.from_iterable(list_of_lists))


def segment_response(text, threshold=LONG_PARAGRAPH_THRESHOLD, min_step=MIN_STEP_LENGTH):
    """Master segmentation pipeline: text -> list of atomic step strings."""
    if not text or not text.strip():
        return []

    # Phase 0: Protect LaTeX
    protected, latex_map = protect_latex(text)

    # Phase 1: Markdown headers
    chunks = split_markdown_headers(protected)

    # Phase 2: Double newlines
    chunks = _flatten(split_double_newline(c) for c in chunks)

    # Phase 3: Display math isolation
    chunks = _flatten(split_display_math(c) for c in chunks)

    # Phase 4: Numbered lists and bullets
    chunks = _flatten(split_numbered_and_bullets(c) for c in chunks)

    # Phase 5: Logical connectors (long chunks only)
    chunks = _flatten(split_logical_connectors(c, threshold) for c in chunks)

    # Phase 6: Sentence split (still-long chunks only)
    chunks = _flatten(split_sentences(c, threshold) for c in chunks)

    # Phase 7: Restore LaTeX and clean up
    chunks = restore_latex(chunks, latex_map)

    # Merge display math blocks with their preceding step
    steps = merge_display_math(chunks)

    # Final cleanup: strip, remove empty, merge short fragments
    steps = postprocess(steps, min_step)

    return steps


# ---------------------------------------------------------------------------
# Record processing & CLI
# ---------------------------------------------------------------------------

VARIANTS = ["original", "simple", "hard"]


def process_record(record, threshold=LONG_PARAGRAPH_THRESHOLD, min_step=MIN_STEP_LENGTH):
    """Segment the response field of each sample in each variant."""
    for variant in VARIANTS:
        entry = record.get(variant)
        if not entry or "samples" not in entry:
            continue
        for sample in entry["samples"]:
            if "response" not in sample:
                continue
            steps = segment_response(sample["response"], threshold, min_step)
            sample["steps"] = [{"index": i + 1, "text": s} for i, s in enumerate(steps)]
            sample["num_steps"] = len(steps)
    return record


def main():
    parser = argparse.ArgumentParser(description="CoT 推理链启发式切分器")
    parser.add_argument("--input", type=str, required=True, help="输入 JSONL 路径")
    parser.add_argument("--output", type=str, default=None, help="输出 JSONL 路径（默认按输入自动推导）")
    parser.add_argument("--threshold", type=int, default=LONG_PARAGRAPH_THRESHOLD,
                        help="长段落阈值（字符数），超过此值才进行逻辑词/句号切分")
    parser.add_argument("--min-step", type=int, default=MIN_STEP_LENGTH,
                        help="最短步骤长度（字符数），过短片段将被合并")
    args = parser.parse_args()

    input_path = args.input
    if args.output:
        output_path = args.output
    else:
        input_path_obj = Path(input_path)
        output_name = input_path_obj.name.replace("all_records", "segmented_records")
        output_path = str(input_path_obj.with_name(output_name))

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    total_steps = 0
    total_samples = 0
    for record in data:
        process_record(record, args.threshold, args.min_step)
        for v in VARIANTS:
            samples = record.get(v, {}).get("samples", [])
            for sample in samples:
                total_steps += sample.get("num_steps", 0)
                total_samples += 1

    with open(output_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    avg_steps = total_steps / total_samples if total_samples else 0
    print(f"已处理 {len(data)} 条记录（{total_samples} 个 response）")
    print(f"共切分出 {total_steps} 个步骤，平均每个 response {avg_steps:.1f} 步")
    print(f"结果已保存至 {output_path}")


if __name__ == "__main__":
    main()
