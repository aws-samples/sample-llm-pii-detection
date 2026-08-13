"""Bedrock-backed Inferencer adapter."""
from pii_detector.inferencer import ContextLengthExceeded
from pii_detector.bedrock import BedrockConverse


# Substring match against the Bedrock error string. Lowercased so a
# capitalisation drift on the AWS side doesn't slip through — the
# comparison `.lower()`s the incoming text.
_BEDROCK_OVERFLOW_MARKER = "exceeds model's maximum context length"


class BedrockInferencer:
    """Wraps :class:`BedrockConverse` to satisfy the ``Inferencer`` protocol.

    Credentials are resolved by boto3's default chain (environment variables,
    the default profile in ``~/.aws/credentials``, or an attached IAM role);
    set ``AWS_PROFILE`` in the environment to target a specific account.
    """

    def __init__(
        self,
        model_id: str,
        region: str = "us-east-1",
        max_tokens: int = 4096,
    ):
        self._llm = BedrockConverse(
            region=region,
            model_id=model_id,
        )
        self._max_tokens = max_tokens
        # Per-request sampling overrides. None entries mean "leave to the model
        # default" — `temperature`/`top_p` ride Converse `inferenceConfig`,
        # `repetition_penalty`/`seed` ride `additionalModelRequestFields` (only
        # some model families honour them; others reject with a
        # ValidationException).
        self._temperature: float | None = None
        self._top_p: float | None = None
        self._repetition_penalty: float | None = None
        self._seed: int | None = None

    @property
    def model_id(self) -> str:
        return self._llm.model_id

    def set_sampling(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
        seed: int | None = None,
    ) -> None:
        """Set the sampling overrides applied on the next ``__call__``.

        Any argument may be ``None`` to leave that knob at the model default.
        Not safe to share a single instance across threads — construct one
        ``BedrockInferencer`` per thread if you need concurrent sampling.

        ``seed`` is not a Converse ``inferenceConfig`` field; like
        ``repetition_penalty`` it rides ``additionalModelRequestFields`` and is
        honoured only by model families whose native request body accepts it
        (e.g. OpenAI gpt-oss, Mistral, DeepSeek). Claude/Nova reject it, and the
        ValidationException surfaces to the caller at request time.
        """
        self._temperature = temperature
        self._repetition_penalty = repetition_penalty
        self._top_p = top_p
        self._seed = seed

    def __call__(self, messages: list[dict]) -> str:
        inference_config: dict = {"maxTokens": self._max_tokens}
        if self._temperature is not None:
            inference_config["temperature"] = self._temperature
        # `topP` is a first-class Converse inferenceConfig field (unlike
        # repetition_penalty). Only sent when set.
        if self._top_p is not None:
            inference_config["topP"] = self._top_p

        # `repetition_penalty` and `seed` are not `inferenceConfig` fields; they
        # are forwarded raw via `additionalModelRequestFields`. Only some model
        # families honour them (Claude/Nova reject both) — the error surfaces to
        # the caller.
        kwargs: dict = {}
        extra_fields: dict = {}
        if self._repetition_penalty is not None:
            extra_fields["repetition_penalty"] = self._repetition_penalty
        if self._seed is not None:
            extra_fields["seed"] = self._seed
        if extra_fields:
            kwargs["additionalModelRequestFields"] = extra_fields

        try:
            raw = self._llm(
                messages=messages,
                inference_config=inference_config,
                **kwargs,
            )
        except Exception as e:
            if _BEDROCK_OVERFLOW_MARKER in str(e).lower():
                raise ContextLengthExceeded(str(e)) from e
            raise
        chunks = [c["text"] for c in raw.get("content", []) if "text" in c]
        return chunks[0] if chunks else ""
