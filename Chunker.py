"""
Chunker.py
----------
Semantic + Recursive document chunker for the RAG ingestion pipeline.

Strategy  : Hybrid (Semantic grouping + Recursive paragraph→sentence split)
Max tokens: 512  (CHUNK_SIZE from config)
Overlap   : 50 tokens (CHUNK_OVERLAP from config)
Special   : Code blocks kept intact (type=code), Tables kept intact (type=table)
Output    : List[dict]  →  [{"text": "...", "type": "text|code|table"}, ...]
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass, field
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Pull constants from project config; fall back to safe defaults if needed
# ---------------------------------------------------------------------------
try:
    from config import CHUNK_SIZE, CHUNK_OVERLAP
except ImportError:
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50

MAX_TOKENS: int = CHUNK_SIZE          # hard upper bound
TARGET_MIN: int = int(MAX_TOKENS * 0.78)  # ~400 tokens
OVERLAP_TOKENS: int = CHUNK_OVERLAP   # tokens carried over between chunks


# ---------------------------------------------------------------------------
# Token counting (whitespace split — fast, no external deps)
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Approximate token count using whitespace splitting."""
    return len(text.split())


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    text: str
    type: str = "text"   # "text" | "code" | "table"

    def to_dict(self) -> dict:
        return {"text": self.text, "type": self.type}


# ---------------------------------------------------------------------------
# Low-level text splitters
# ---------------------------------------------------------------------------

def _split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using a regex that respects common abbreviations.
    Returns a list of non-empty sentence strings.
    """
    # Split on sentence-ending punctuation followed by whitespace / end-of-str
    sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')
    parts = sentence_endings.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_into_paragraphs(text: str) -> List[str]:
    """Split by one or more blank lines."""
    return [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]


def _sentence_overlap(sentences: List[str], n_tokens: int) -> str:
    """
    Return the last `n_tokens` tokens worth of sentences from a list.
    Used to create leading overlap for the next chunk.
    """
    overlap_parts: List[str] = []
    running = 0
    for sent in reversed(sentences):
        t = _token_count(sent)
        if running + t > n_tokens:
            break
        overlap_parts.insert(0, sent)
        running += t
    return " ".join(overlap_parts)


# ---------------------------------------------------------------------------
# Code-block extractor
# ---------------------------------------------------------------------------

def _extract_code_blocks(text: str) -> List[Tuple[str, str]]:
    """
    Extract fenced code blocks (``` ... ```) from text.

    Returns a list of (placeholder, code_block_text) tuples.
    The placeholder can be used to split the surrounding text while preserving
    positions.
    """
    pattern = re.compile(r'(```[\s\S]*?```)', re.MULTILINE)
    blocks = pattern.findall(text)
    placeholders = []
    for i, block in enumerate(blocks):
        placeholder = f"__CODE_BLOCK_{i}__"
        placeholders.append((placeholder, block))
        text = text.replace(block, placeholder, 1)
    return text, placeholders


# ---------------------------------------------------------------------------
# Table extractor (Markdown tables)
# ---------------------------------------------------------------------------

def _extract_tables(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Extract Markdown tables from text, replacing them with placeholders.
    A Markdown table is: one or more lines containing |, with a separator row.
    """
    # Match blocks of lines that look like a markdown table
    pattern = re.compile(
        r'((?:\|[^\n]*\|\n)+(?:\|[-: |]+\|\n)(?:\|[^\n]*\|\n?)*)',
        re.MULTILINE
    )
    tables = pattern.findall(text)
    placeholders = []
    for i, table in enumerate(tables):
        placeholder = f"__TABLE_BLOCK_{i}__"
        placeholders.append((placeholder, table.strip()))
        text = text.replace(table, placeholder, 1)
    return text, placeholders


# ---------------------------------------------------------------------------
# Core chunking logic for plain text
# ---------------------------------------------------------------------------

def _chunk_plain_text(text: str, pending_overlap: str = "") -> List[Chunk]:
    """
    Recursively chunk plain text using paragraph → sentence → word priority.
    Respects MAX_TOKENS and injects overlap from the previous chunk.
    """
    chunks: List[Chunk] = []
    paragraphs = _split_into_paragraphs(text)

    current_sentences: List[str] = []
    current_tokens: int = 0

    # If we have overlap from the previous chunk, seed current buffer with it
    if pending_overlap:
        overlap_tokens = _token_count(pending_overlap)
        if overlap_tokens < MAX_TOKENS:
            current_sentences = [pending_overlap]
            current_tokens = overlap_tokens

    def _flush(sents: List[str]) -> Tuple[List[str], int]:
        """Emit a chunk and return the overlap seed for the next chunk."""
        chunk_text = " ".join(sents).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, type="text"))
        overlap_text = _sentence_overlap(sents, OVERLAP_TOKENS)
        overlap_toks = _token_count(overlap_text)
        return ([overlap_text] if overlap_text else []), overlap_toks

    for para in paragraphs:
        para_sentences = _split_into_sentences(para)

        for sent in para_sentences:
            sent_tokens = _token_count(sent)

            # Single sentence exceeds MAX — must hard-split at word level
            if sent_tokens > MAX_TOKENS:
                # Flush whatever is buffered first
                if current_sentences:
                    current_sentences, current_tokens = _flush(current_sentences)

                words = sent.split()
                word_buf: List[str] = []
                word_count = 0
                for word in words:
                    if word_count + 1 > MAX_TOKENS:
                        chunk_text = " ".join(word_buf)
                        chunks.append(Chunk(text=chunk_text, type="text"))
                        # overlap: last OVERLAP_TOKENS words
                        overlap_words = word_buf[-OVERLAP_TOKENS:]
                        word_buf = overlap_words + [word]
                        word_count = len(word_buf)
                    else:
                        word_buf.append(word)
                        word_count += 1
                if word_buf:
                    # leave as residual in current_sentences
                    residual = " ".join(word_buf)
                    current_sentences = [residual]
                    current_tokens = _token_count(residual)
                continue

            # Adding this sentence would exceed the hard limit → flush first
            if current_tokens + sent_tokens > MAX_TOKENS:
                current_sentences, current_tokens = _flush(current_sentences)

            current_sentences.append(sent)
            current_tokens += sent_tokens

            # If we've hit the target range, flush proactively for clean chunks
            if current_tokens >= TARGET_MIN:
                current_sentences, current_tokens = _flush(current_sentences)

    # Flush remaining buffer
    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, type="text"))

    return chunks


# ---------------------------------------------------------------------------
# Table chunker (keeps rows together; repeats header in continuations)
# ---------------------------------------------------------------------------

def _chunk_table(table_text: str) -> List[Chunk]:
    """
    Chunk a Markdown table.  Never splits a row.
    Repeats the header + separator rows in continuation chunks.
    """
    lines = [ln for ln in table_text.splitlines() if ln.strip()]
    if not lines:
        return []

    # Identify header: first line, separator: second line (--- pattern)
    header_lines: List[str] = []
    data_lines: List[str] = []

    if len(lines) >= 2 and re.match(r'\|[-: |]+\|', lines[1]):
        header_lines = lines[:2]   # header row + separator
        data_lines = lines[2:]
    else:
        data_lines = lines

    header_text = "\n".join(header_lines)
    header_tokens = _token_count(header_text)

    chunks: List[Chunk] = []
    current_rows: List[str] = list(header_lines)
    current_tokens: int = header_tokens

    for row in data_lines:
        row_tokens = _token_count(row)

        if current_tokens + row_tokens > MAX_TOKENS:
            # Flush
            chunk_text = "\n".join(current_rows).strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, type="table"))
            # Start new chunk with header repeated
            current_rows = list(header_lines) + [row]
            current_tokens = header_tokens + row_tokens
        else:
            current_rows.append(row)
            current_tokens += row_tokens

    if current_rows:
        chunk_text = "\n".join(current_rows).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, type="table"))

    return chunks


# ---------------------------------------------------------------------------
# Code block chunker
# ---------------------------------------------------------------------------

def _chunk_code(code_text: str) -> List[Chunk]:
    """
    Keep an entire code block in one chunk.
    If it exceeds MAX_TOKENS, split at line boundaries (last resort).
    """
    if _token_count(code_text) <= MAX_TOKENS:
        return [Chunk(text=code_text.strip(), type="code")]

    # Split by lines, keeping ``` fences
    lines = code_text.splitlines()
    chunks: List[Chunk] = []
    current_lines: List[str] = []
    current_tokens = 0
    in_fence = False

    for line in lines:
        line_tokens = _token_count(line)
        if current_tokens + line_tokens > MAX_TOKENS and current_lines:
            # Close fence if open
            block = "\n".join(current_lines)
            if in_fence and not block.rstrip().endswith("```"):
                block += "\n```"
            chunks.append(Chunk(text=block.strip(), type="code"))
            # Start new with opening fence
            current_lines = ["```"]
            current_tokens = 1
            in_fence = True

        current_lines.append(line)
        current_tokens += line_tokens

        if line.strip().startswith("```"):
            in_fence = not in_fence

    if current_lines:
        chunks.append(Chunk(text="\n".join(current_lines).strip(), type="code"))

    return chunks


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def chunk_document(text: str) -> List[dict]:
    """
    Chunk a raw document string into semantically meaningful pieces.

    Parameters
    ----------
    text : str
        Raw document text (may contain code blocks and Markdown tables).

    Returns
    -------
    List[dict]
        Each dict has keys ``"text"`` and ``"type"`` ("text" | "code" | "table").
    """
    if not text or not text.strip():
        return []

    # ── Step 1: extract special blocks, replace with placeholders ──────────
    text, code_placeholders = _extract_code_blocks(text)
    text, table_placeholders = _extract_tables(text)

    all_placeholders: dict[str, List[Chunk]] = {}

    for placeholder, code_block in code_placeholders:
        all_placeholders[placeholder] = _chunk_code(code_block)

    for placeholder, table_block in table_placeholders:
        all_placeholders[placeholder] = _chunk_table(table_block)

    # ── Step 2: split remaining text into segments around placeholders ──────
    # Build a regex that matches any placeholder
    if all_placeholders:
        ph_pattern = re.compile(
            "(" + "|".join(re.escape(k) for k in all_placeholders) + ")"
        )
        segments = ph_pattern.split(text)
    else:
        segments = [text]

    # ── Step 3: process each segment in order ──────────────────────────────
    result_chunks: List[Chunk] = []
    pending_overlap: str = ""

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        if segment in all_placeholders:
            # Special block — emit its chunks; reset overlap
            special_chunks = all_placeholders[segment]
            result_chunks.extend(special_chunks)
            pending_overlap = ""   # Special blocks break overlap continuity
        else:
            # Plain text — chunk with carry-over overlap
            text_chunks = _chunk_plain_text(segment, pending_overlap)
            if text_chunks:
                result_chunks.extend(text_chunks)
                # Update overlap from last plain text chunk
                last_text = text_chunks[-1].text
                pending_overlap = _sentence_overlap(
                    _split_into_sentences(last_text), OVERLAP_TOKENS
                )

    # ── Step 4: final safety — hard-split any chunk that still exceeds limit ─
    safe_chunks: List[Chunk] = []
    for chunk in result_chunks:
        if _token_count(chunk.text) > MAX_TOKENS:
            # Force word-level split as absolute last resort
            words = chunk.text.split()
            buf: List[str] = []
            for word in words:
                if _token_count(" ".join(buf) + " " + word) > MAX_TOKENS:
                    safe_chunks.append(Chunk(text=" ".join(buf), type=chunk.type))
                    buf = buf[-OVERLAP_TOKENS:] + [word]
                else:
                    buf.append(word)
            if buf:
                safe_chunks.append(Chunk(text=" ".join(buf), type=chunk.type))
        else:
            safe_chunks.append(chunk)

    return [c.to_dict() for c in safe_chunks if c.text.strip()]


# ---------------------------------------------------------------------------
# Convenience: chunk from JSON prompt format  (used by the pipeline)
# ---------------------------------------------------------------------------

def chunk_from_prompt(prompt_json: str) -> str:
    """
    Accept the JSON prompt format used in the system prompt:
      {"input_text": "..."}
    Returns a JSON string: [{"text": "...", "type": "..."}, ...]
    """
    data = json.loads(prompt_json)
    input_text = data.get("input_text", "")
    chunks = chunk_document(input_text)
    return json.dumps(chunks, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point  —  python Chunker.py <file.txt>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python Chunker.py <path_to_document>")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    result = chunk_document(raw)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n── Total chunks: {len(result)} ──", file=sys.stderr)
