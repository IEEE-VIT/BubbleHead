from __future__ import annotations
import re
import json
import logging
from dataclasses import dataclass
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
# Logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tiktoken encoder (cl100k_base covers GPT-4 / text-embedding-3 tokenization;
# a close enough proxy for BPE counts on code-heavy documents)
# ---------------------------------------------------------------------------
_ENCODER = None


def _get_encoder():
    """Lazy-load the tiktoken encoder on first use."""
    global _ENCODER
    if _ENCODER is None:
        import tiktoken
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def _token_count(text: str) -> int:
    """Return the number of BPE tokens in *text* using tiktoken (cl100k_base)."""
    return len(_get_encoder().encode(text))


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

def _extract_code_blocks(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Extract fenced code blocks (``` ... ```) from text.

    Returns a tuple of:
      - the original text with each code block replaced by a placeholder string
      - a list of (placeholder, code_block_text) pairs
    """
    pattern = re.compile(r'(```[\s\S]*?```)', re.MULTILINE)
    blocks = pattern.findall(text)
    placeholders: List[Tuple[str, str]] = []
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

    Matches any block of pipe-delimited lines that contains at least one
    separator row (cells made up of dashes, colons, and spaces). The
    separator row may appear anywhere in the block, which handles tables
    whose header row is preceded by a caption line or is omitted entirely.
    """
    # Each line must contain at least one pipe character; the block must
    # include at least one separator row of the form |---|---|.
    pattern = re.compile(
        r'((?:[^\n]*\|[^\n]*\n)*'   # zero or more leading pipe lines
        r'[^\n]*\|[-: |]+\|[^\n]*'  # the separator row (required)
        r'(?:\n[^\n]*\|[^\n]*)*)',   # zero or more trailing pipe lines
        re.MULTILINE,
    )
    tables = pattern.findall(text)
    placeholders: List[Tuple[str, str]] = []
    for i, table in enumerate(tables):
        stripped = table.strip()
        if not stripped:
            continue
        placeholder = f"__TABLE_BLOCK_{i}__"
        placeholders.append((placeholder, stripped))
        text = text.replace(table, placeholder, 1)
    return text, placeholders


# ---------------------------------------------------------------------------
# Core chunking logic for plain text
# ---------------------------------------------------------------------------

def _chunk_plain_text(text: str, pending_overlap: str = "") -> List[Chunk]:
    """
    Chunk plain text using paragraph -> sentence -> word priority.
    Respects MAX_TOKENS and injects overlap from the previous chunk.

    The inner ``_flush`` helper is intentionally kept pure: it receives the
    current sentence buffer, emits a Chunk into ``chunks``, and returns the
    new (overlap_buffer, overlap_token_count) tuple so the caller can update
    its own state explicitly.  This avoids the confusion of a closure that
    both appends side-effects and returns values.
    """
    chunks: List[Chunk] = []
    paragraphs = _split_into_paragraphs(text)

    current_sentences: List[str] = []
    current_tokens: int = 0

    if pending_overlap:
        overlap_tokens = _token_count(pending_overlap)
        if overlap_tokens < MAX_TOKENS:
            current_sentences = [pending_overlap]
            current_tokens = overlap_tokens

    def _flush(sents: List[str]) -> Tuple[List[str], int]:
        """
        Emit a Chunk for *sents* and return the seed for the next chunk as
        ``([overlap_text], overlap_token_count)``.  The caller is responsible
        for replacing its own ``current_sentences`` / ``current_tokens`` with
        the returned values.
        """
        chunk_text = " ".join(sents).strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, type="text"))
            logger.debug("Flushed text chunk (%d tokens)", _token_count(chunk_text))
        overlap_text = _sentence_overlap(sents, OVERLAP_TOKENS)
        overlap_toks = _token_count(overlap_text)
        return ([overlap_text] if overlap_text else []), overlap_toks

    for para in paragraphs:
        para_sentences = _split_into_sentences(para)

        for sent in para_sentences:
            sent_tokens = _token_count(sent)

            # Single sentence exceeds MAX - must hard-split at word level
            if sent_tokens > MAX_TOKENS:
                if current_sentences:
                    current_sentences, current_tokens = _flush(current_sentences)
                
                words = sent.split()
                word_buf: List[str] = []
                word_buf_tokens = 0

                for word in words:
                    # Calculate tokens for buffer with this word added
                    test_text = " ".join(word_buf + [word])
                    test_tokens = _token_count(test_text)
                    
                    if test_tokens > MAX_TOKENS and word_buf:
                        # Flush current buffer
                        chunk_text = " ".join(word_buf)
                        chunks.append(Chunk(text=chunk_text, type="text"))
                        logger.debug("Hard word-split chunk (%d tokens)", word_buf_tokens)
                        
                        # Keep last OVERLAP_TOKENS worth of words as overlap
                        overlap_words = []
                        for w in reversed(word_buf):
                            test_overlap = " ".join([w] + overlap_words)
                            if _token_count(test_overlap) > OVERLAP_TOKENS:
                                break
                            overlap_words.insert(0, w)

                        word_buf = overlap_words + [word]
                        word_buf_tokens = _token_count(" ".join(word_buf))
                    else:
                        word_buf.append(word)
                        word_buf_tokens = test_tokens
                
                if word_buf:
                    residual = " ".join(word_buf)
                    current_sentences = [residual]
                    current_tokens = _token_count(residual)
                continue

            if current_tokens + sent_tokens > MAX_TOKENS:
                current_sentences, current_tokens = _flush(current_sentences)

            current_sentences.append(sent)
            current_tokens += sent_tokens

            if current_tokens >= TARGET_MIN:
                current_sentences, current_tokens = _flush(current_sentences)

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

    header_lines: List[str] = []
    data_lines: List[str] = []

    if len(lines) >= 2 and re.match(r'\|[-: |]+\|', lines[1]):
        header_lines = lines[:2]
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
        
        # Account for newline token when adding row
        test_text = "\n".join(current_rows + [row])
        test_tokens = _token_count(test_text)

        if test_tokens > MAX_TOKENS and len(current_rows) > len(header_lines):
            # Flush current chunk
            chunk_text = "\n".join(current_rows).strip()
            if chunk_text:
                chunks.append(Chunk(text=chunk_text, type="table"))
                logger.debug("Flushed table chunk (%d tokens)", current_tokens)
            # Start new chunk with header + current row
            current_rows = list(header_lines) + [row]
            current_tokens = _token_count("\n".join(current_rows))
        else:
            current_rows.append(row)
            current_tokens = test_tokens

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

    logger.warning(
        "Code block exceeds MAX_TOKENS (%d); splitting at line boundaries.",
        MAX_TOKENS,
    )

    lines = code_text.splitlines()
    chunks: List[Chunk] = []
    current_lines: List[str] = []
    current_tokens = 0
    in_fence = False

    for line in lines:
        line_tokens = _token_count(line)
        
        # Test if adding this line would exceed limit
        test_lines = current_lines + [line]
        test_text = "\n".join(test_lines)
        test_tokens = _token_count(test_text)
        
        if test_tokens > MAX_TOKENS and current_lines:
            # Flush current block
            block = "\n".join(current_lines)
            if in_fence and not block.rstrip().endswith("```"):
                block += "\n```"
            chunks.append(Chunk(text=block.strip(), type="code"))
            # Start new block with fence marker
            current_lines = ["```"]
            current_tokens = _token_count("```")
            in_fence = True
            # Recalculate with new line
            test_lines = current_lines + [line]
            test_tokens = _token_count("\n".join(test_lines))

        current_lines.append(line)
        current_tokens = test_tokens

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

    logger.info("Starting document chunking (%d chars)", len(text))

    # Step 1: extract special blocks, replace with placeholders
    text, code_placeholders = _extract_code_blocks(text)
    text, table_placeholders = _extract_tables(text)

    all_placeholders: dict[str, List[Chunk]] = {}

    for placeholder, code_block in code_placeholders:
        all_placeholders[placeholder] = _chunk_code(code_block)

    for placeholder, table_block in table_placeholders:
        all_placeholders[placeholder] = _chunk_table(table_block)

    # Step 2: split remaining text into segments around placeholders
    if all_placeholders:
        ph_pattern = re.compile(
            "(" + "|".join(re.escape(k) for k in all_placeholders) + ")"
        )
        segments = ph_pattern.split(text)
    else:
        segments = [text]

    # Step 3: process each segment in order
    result_chunks: List[Chunk] = []
    pending_overlap: str = ""

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        if segment in all_placeholders:
            special_chunks = all_placeholders[segment]
            result_chunks.extend(special_chunks)
            pending_overlap = ""
        else:
            text_chunks = _chunk_plain_text(segment, pending_overlap)
            if text_chunks:
                result_chunks.extend(text_chunks)
                last_text = text_chunks[-1].text
                pending_overlap = _sentence_overlap(
                    _split_into_sentences(last_text), OVERLAP_TOKENS
                )

    # Step 4: final safety - hard-split any chunk that still exceeds limit
    safe_chunks: List[Chunk] = []
    for chunk in result_chunks:
        if _token_count(chunk.text) > MAX_TOKENS:
            logger.warning(
                "Chunk still over limit after primary pass; applying word-level fallback."
            )
            words = chunk.text.split()
            buf: List[str] = []
            buf_tokens = 0
            
            for word in words:
                # Calculate tokens for buffer with this word added
                test_text = " ".join(buf + [word])
                test_tokens = _token_count(test_text)
                
                if test_tokens > MAX_TOKENS and buf:
                    # Flush current buffer
                    safe_chunks.append(Chunk(text=" ".join(buf), type=chunk.type))
                    
                    # Keep last OVERLAP_TOKENS worth of words as overlap
                    overlap_words = []
                    for w in reversed(buf):
                        test_overlap = " ".join([w] + overlap_words)
                        if _token_count(test_overlap) > OVERLAP_TOKENS:
                            break
                        overlap_words.insert(0, w)

                    buf = overlap_words + [word]
                    buf_tokens = _token_count(" ".join(buf))
                else:
                    buf.append(word)
                    buf_tokens = test_tokens
                    
            if buf:
                safe_chunks.append(Chunk(text=" ".join(buf), type=chunk.type))
        else:
            safe_chunks.append(chunk)

    final = [c.to_dict() for c in safe_chunks if c.text.strip()]
    logger.info("Chunking complete - %d chunks produced", len(final))
    return final


# ---------------------------------------------------------------------------
# Convenience: chunk from JSON prompt format  (used by the pipeline)
# ---------------------------------------------------------------------------

def chunk_from_prompt(prompt_json: str) -> str:
    """
    Thin wrapper used by the ingestion pipeline's LangGraph node.

    The pipeline passes documents to this module as a JSON string matching
    the schema ``{"input_text": "<raw document text>"}``.  The function
    chunks the text and returns the result as a JSON string
    ``[{"text": "...", "type": "..."}, ...]`` so the node can deserialise it
    and forward the chunks to the embedding step without needing to import
    ``chunk_document`` directly.

    Example
    -------
    >>> chunk_from_prompt('{"input_text": "Hello world."}')
    '[{"text": "Hello world.", "type": "text"}]'
    """
    data = json.loads(prompt_json)
    input_text = data.get("input_text", "")
    chunks = chunk_document(input_text)
    return json.dumps(chunks, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# CLI entry point  -  python Chunker.py <file.txt>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    if len(sys.argv) < 2:
        print("Usage: python Chunker.py <path_to_document>")
        sys.exit(1)

    file_path = sys.argv[1]
    with open(file_path, "r", encoding="utf-8") as f:
        raw = f.read()

    result = chunk_document(raw)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    logger.info("Total chunks: %d", len(result))