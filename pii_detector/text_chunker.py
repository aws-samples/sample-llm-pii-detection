"""Word-based text chunking for the PII detector.

``chunk_text`` splits a long input into non-overlapping, offset-tracked windows
so the detector can run inference per chunk and shift each chunk's local offsets
back into the original coordinate space. Windows are measured in CHARACTERS
(this package has no tokenizer).

Chunks are grown a word at a time until adding the next word would exceed
``max_chars``; a single word longer than ``max_chars`` is split character-wise.
Each chunk is a real slice of the original text (``text[start:end]``), so its
``char_offset`` maps chunk-relative detection offsets back exactly — no
whitespace is collapsed.
"""
import re


# Proactive-chunking default. 40k chars (~10k tokens) keeps the window inside
# the regime where records actually succeed: longer input correlates with the
# model silently missing PII.
CHUNK_MAX_CHARS = 40000

# Floor below which a chunk that STILL overflows the backend is treated as
# un-splittable: a backend rejecting even a small chunk won't be fixed by more
# slicing, so the detector drops it rather than recursing down to 1-char windows.
MIN_CHUNK_CHARS = 400


def _tokens(text: str, max_chars: int):
    """Yield ``(start, end)`` spans of each whitespace-delimited word, breaking
    any word longer than ``max_chars`` into consecutive character-wise spans so
    no yielded span exceeds ``max_chars``."""
    for m in re.finditer(r"\S+", text):
        start, end = m.start(), m.end()
        if end - start <= max_chars:
            yield start, end
        else:
            for i in range(start, end, max_chars):
                yield i, min(i + max_chars, end)


def chunk_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> list[tuple[str, int]]:
    """Split ``text`` into non-overlapping ``(chunk, char_offset)`` slices.

    ``char_offset`` is where ``chunk`` begins in ``text``, so a detection found
    at chunk-relative position ``p`` maps to ``char_offset + p`` in the original.
    Returns an empty list for empty or whitespace-only input.
    """
    max_chars = max(1, max_chars)
    chunks: list[tuple[str, int]] = []
    cur_start: int | None = None
    cur_end = 0
    for start, end in _tokens(text, max_chars):
        if cur_start is None:
            # Start a fresh chunk on the first (or post-flush) token.
            cur_start, cur_end = start, end
        elif end - cur_start <= max_chars:
            # Extend the current chunk to include this word (and the whitespace
            # between it and the previous word — the slice preserves it).
            cur_end = end
        else:
            chunks.append((text[cur_start:cur_end], cur_start))
            cur_start, cur_end = start, end
    if cur_start is not None:
        chunks.append((text[cur_start:cur_end], cur_start))
    return chunks
