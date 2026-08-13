"""Inferencer protocol for PiiDetector.

PiiDetector is backend-agnostic: any object with the ``Inferencer`` shape
(``__call__(messages) -> str`` plus a ``model_id`` property) can drive it.

Adapters are responsible for translating their backend's context-overflow
error to ``ContextLengthExceeded`` so PiiDetector has a single signal to
catch when it needs to split the input.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Inferencer(Protocol):
    """Backend abstraction used by ``PiiDetector``.

    ``messages`` follows the Bedrock Converse shape:
        ``[{"role": "user", "content": [{"text": "..."}]}]``
    The return value is the assistant's plain text — PiiDetector handles
    the JSON parsing.
    """

    def __call__(self, messages: list[dict]) -> str: ...

    @property
    def model_id(self) -> str: ...

    def set_sampling(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Set per-request sampling overrides for the calling thread's next call.

        Keyword-only: callers always pass ``temperature``/``top_p``/
        ``repetition_penalty``/``seed`` by name. ``None`` leaves a knob at the
        backend default. ``repetition_penalty`` and ``seed`` are honoured by
        some model families and rejected by others (e.g. Claude/Nova on Bedrock,
        which have neither in their Converse request body) — the rejection
        surfaces to the caller at request time. Implementations scope this thread-locally so
        a shared/pooled inferencer can serve concurrent requests with different
        sampling. Adapters that accept no sampling may implement it as a no-op."""
        ...


class ContextLengthExceeded(Exception):
    """Raised by an Inferencer when the input exceeds the model's context window."""


__all__ = ["Inferencer", "ContextLengthExceeded"]
