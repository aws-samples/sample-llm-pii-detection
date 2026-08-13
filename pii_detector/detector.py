import json
import logging
import re
from typing import Any

from pii_detector.inferencer import ContextLengthExceeded, Inferencer
from pii_detector.label_recovery import recover_label, valid_label_set
from pii_detector.text_chunker import CHUNK_MAX_CHARS, MIN_CHUNK_CHARS, chunk_text
from pii_detector import templates


logger = logging.getLogger(__name__)


class PiiDetector:
    def __init__(self, inferencer: Inferencer,
                 extra_definitions: dict | None = None,
                 extra_public_drops: list | None = None,
                 extra_examples: list | None = None,
                 model_id: str | None = None):
        self._inferencer = inferencer
        # Prompt source: the templates module shipped in this package. The
        # category lists are rendered once here; the bare system-prompt template
        # is held for per-row `.format` in __call__.
        self._system_prompt = templates.PII_DETECTION_SYSTEM_PROMPT
        self._pii_categories = templates.build_pii_categories(extra_definitions)
        # The exact vocabulary the model is shown (parsed from the rendered
        # category block, so it tracks base vs. injected extras). Used by the
        # recovery pass to re-home misspelled/plural label variants and to
        # recognise what's genuinely off-template.
        self._valid_labels = valid_label_set(self._pii_categories)
        self._public_categories = templates.build_public_categories(extra_public_drops)
        self._example_section = templates.build_example_section(extra_examples)
        self._model_id_override = model_id

    @property
    def model_id(self):
        # Logging label only. Callers pass their display name so logs read
        # cleanly; otherwise fall back to the inferencer's model id.
        return self._model_id_override or f"PiiDetector-{self._inferencer.model_id}"

    @staticmethod
    def read_json_response(response: str, type_str: str):
        if "```json" in response and response.count("```") >= 2:
            response = response[response.index("```json") + len("```json"): response.rindex("```")]
        result: Any
        if type_str == "list":
            result = []
            delimiters = ("[", "]")
        elif type_str == "dict":
            result = {}
            delimiters = ("{", "}")
        else:
            result = None
        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            logger.debug("read_json_response: direct json.loads failed; falling back to bracket-substring parse")
            result = None
        if result is None:
            try:
                result = json.loads(response[response.index(delimiters[0]): response.rindex(delimiters[1]) + 1])
            except json.JSONDecodeError:
                logger.debug("read_json_response: bracket-substring parse failed (JSONDecodeError)")
            except ValueError:
                logger.debug("read_json_response: bracket delimiters not found in response (ValueError)")
        return result

    def recover_labels(self, detection: list) -> list:
        """Re-home misspelled/plural label variants onto the prompt vocabulary.

        Each item's ``pii_entity_type`` is replaced by the recovered valid label
        (e.g. ``DATE`` -> ``DATES``); an emission that no recovery tier can map
        is relabelled ``UNK``. The ORIGINAL emitted string is preserved on
        ``emitted_label`` so downstream reporting can still surface what the
        model actually produced (a true off-template hallucination stays
        visible by its real name, not collapsed into ``UNK``).

        ``emitted_label`` is added unconditionally — even when the label was
        already valid — so the field's presence never implies "was recovered";
        callers compare it against ``pii_entity_type`` to detect a change.

        Items missing ``pii_entity_type`` or ``pii_entity_value`` are dropped:
        a malformed emission can't be located in the text and would otherwise
        ``KeyError`` in ``offset_calculation``, which indexes both keys directly.
        """
        recovered = []
        for d in detection:
            if "pii_entity_type" not in d or "pii_entity_value" not in d:
                # Log the keys only, never the values — a detection item can
                # carry raw PII in pii_entity_value and console logs aren't a
                # PII sink.
                logger.error(f"[PII|{self.model_id}] Dropping malformed detection item, keys: {sorted(d.keys())}")
                continue
            emitted = d["pii_entity_type"]
            new_d = dict(d)
            # recover_label returns UNRECOVERED_LABEL itself when nothing maps.
            new_d["pii_entity_type"] = recover_label(emitted, self._valid_labels)
            new_d["emitted_label"] = emitted
            recovered.append(new_d)
        return recovered

    @staticmethod
    def offset_calculation(input_text: str, detection: list) -> list:
        # `re.finditer` returns every occurrence of the value in the text, so
        # an LLM that emits the same value N times would otherwise produce
        # N×M predictions for M occurrences. Dedupe on (start, end, type) to
        # keep one prediction per unique offset/label triple.
        result = []
        seen: set = set()
        for d in detection:
            pii_value = d["pii_entity_value"]
            pii_type = d["pii_entity_type"]
            pattern = r"[\s]+".join(map(re.escape, pii_value.split()))
            if not pattern:
                continue
            for occurrence in re.finditer(pattern, input_text):
                key = (occurrence.start(), occurrence.end(), pii_type)
                if key in seen:
                    continue
                seen.add(key)
                entry = {
                    "pii_entity_type": pii_type,
                    "pii_entity_value": occurrence.group(),
                    "start": occurrence.start(),
                    "end": occurrence.end()
                }
                # Carry the model's original label through when recovery
                # remapped it, so the report's off-template tally keeps the
                # real emitted string rather than the recovered/UNK label.
                if "emitted_label" in d:
                    entry["emitted_label"] = d["emitted_label"]
                result.append(entry)
        return result

    def __call__(self, text: str) -> list:
        # Chunk the whole input into non-overlapping windows, detect per chunk,
        # and shift each chunk's chunk-relative offsets back into the original
        # coordinate space by its char_offset.
        agg_res = []
        for chunk, char_offset in chunk_text(text):
            for entry in self._detect(chunk):
                entry["start"] += char_offset
                entry["end"] += char_offset
                agg_res.append(entry)
        return agg_res

    def _detect(self, current_text: str) -> list:
        """Run detection on one chunk; returned offsets are chunk-relative.

        Retries up to 5×. A ContextLengthExceeded means even this chunk is too
        large for the backend's true token window (char-based chunking can't see
        token counts), so we recursively split it further with a shrunken window
        and aggregate the sub-chunk offsets back into this chunk's coordinate
        space — the safety net the proactive char-windowing alone can't
        guarantee.
        """
        prompt = self._system_prompt.format(
            pii_categories=self._pii_categories,
            public_categories=self._public_categories,
            example_section=self._example_section,
            conversation=current_text
        )
        messages = [
            {"role": "user", "content": [{"text": prompt}]},
        ]
        final_res = None
        attempt = 0
        while final_res is None and attempt < 5:
            attempt += 1
            resp = None
            # Scope this try to the one call that can raise ContextLengthExceeded
            # — the inference itself. The JSON parsing below can raise its own
            # (unrelated) errors; keeping it out of this block means a parse
            # failure is logged as a parse failure, not masked as an "Inference
            # Error", and the overflow signal is never shadowed.
            try:
                response_text = self._inferencer(messages)
            except ContextLengthExceeded as e:
                logger.error(f"[PII|{self.model_id}] Context overflow: {e}")
                logger.error(f"Conv length: {len(current_text.split())} / {len(current_text)}")
                # A chunk small enough to no longer be meaningfully splittable
                # that STILL overflows is a backend problem no further slicing
                # can fix (e.g. a misconfigured backend that always rejects) —
                # bail rather than recurse all the way down to 1-char windows.
                if len(current_text) <= MIN_CHUNK_CHARS:
                    logger.error(f"[PII|{self.model_id}] Dropping un-splittable overflowing chunk ({len(current_text)} chars)")
                    return []
                # Re-split this chunk with a window 25% smaller and recurse.
                # Sub-chunk offsets are relative to current_text — this method's
                # coordinate space — so shifting them by each sub-chunk's offset
                # lands them correctly, and the outer caller's shift then
                # composes on top.
                sub_max = min(int(len(current_text) * 0.75), CHUNK_MAX_CHARS)
                agg_res = []
                for chunk, char_offset in chunk_text(current_text, max_chars=sub_max):
                    for entry in self._detect(chunk):
                        entry["start"] += char_offset
                        entry["end"] += char_offset
                        agg_res.append(entry)
                return agg_res
            except Exception as e:
                logger.error(f"[PII|{self.model_id}] Inference Error. {e.__class__.__name__}: {e}")
                logger.error(f"Conv length: {len(current_text.split())} / {len(current_text)}")
                continue

            if response_text:
                try:
                    resp = self.read_json_response(response_text, "list")
                except Exception as e:
                    logger.error(f"[PII|{self.model_id}] Response Parse Error. {e.__class__.__name__}: {e}")

            if resp is not None:
                try:
                    final_res = self.offset_calculation(current_text, self.recover_labels(resp))
                except Exception as e:
                    logger.error(f"[PII|{self.model_id}] Offset Calculation Error. {e.__class__.__name__}: {e}")
        return final_res if final_res is not None else []
