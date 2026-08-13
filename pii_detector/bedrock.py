"""Bedrock Converse client.

A thin wrapper over the ``bedrock-runtime`` Converse API with a retry/throttle
loop. Credentials come from boto3's default resolution chain (environment
variables, the default profile in ``~/.aws/credentials``, or an attached IAM
role) — set ``AWS_PROFILE`` / ``AWS_REGION`` in the environment to point it at a
specific account.
"""
import boto3
import logging

from time import sleep

from botocore.config import Config


logger = logging.getLogger(__name__)


class BedrockConverse:
    """Client for interactions through the Bedrock Converse API."""

    N_ATTEMPTS = 4
    # boto's default read_timeout is 60s — too short for a long generation
    # (max_tokens up to 4096) under concurrency, which surfaces as a
    # ReadTimeoutError and, after N_ATTEMPTS with no backoff, a fatal
    # RuntimeError. 300s is a generous ceiling for a long generation. botocore's
    # own retries are disabled (max_attempts=0) because invoke() already owns the
    # retry loop; leaving them on would multiply attempts against N_ATTEMPTS.
    READ_TIMEOUT = 300
    CONNECT_TIMEOUT = 10
    CONTENT_FILTER_MSGS = {
        "this request has been blocked by our content filters",
        "content filters",
    }
    BLOCKED = "<blocked by CM>"
    THROTTLE_SLEEP = 5

    def __init__(
        self,
        region: str,
        model_id: str,
        boto_client=None,
    ):
        self.region = region
        if boto_client is not None:
            self.br_client = boto_client
        else:
            self.br_client = self._instantiate_client()
        self.model_id = model_id

    def _instantiate_client(self):
        return boto3.Session().client(
            service_name="bedrock-runtime", region_name=self.region,
            config=Config(
                read_timeout=self.READ_TIMEOUT,
                connect_timeout=self.CONNECT_TIMEOUT,
                retries={"max_attempts": 0},
            ),
        )

    def invoke(self, **kwargs):
        """Call Converse with a retry/throttle loop.

        Retries transient failures: content-filter blocks resolve to the
        ``{"response": BLOCKED}`` sentinel, expired tokens re-instantiate the
        session, and throttling backs off ``THROTTLE_SLEEP`` seconds before the
        next attempt. Exhausting ``N_ATTEMPTS`` raises ``RuntimeError``.
        """
        response = None
        n_attempts = 0
        last_exception = None
        while response is None and n_attempts < self.N_ATTEMPTS:
            n_attempts += 1
            try:
                response = self.br_client.converse(**kwargs)
            except Exception as e:
                if any(msg in str(e).lower() for msg in self.CONTENT_FILTER_MSGS):
                    logger.debug("Request blocked by content moderation")
                    response = {"response": self.BLOCKED}
                elif "Expired" in str(e):
                    logger.warning("Expired token, resetting session")
                    self.br_client = self._instantiate_client()
                elif "ThrottlingException" in str(e) or "ServiceUnavailableException" in str(e) or "Too many requests to model" in str(e):
                    logger.warning(
                        f"[{self.model_id}] Throttling exception, waiting for {self.THROTTLE_SLEEP} seconds..."
                    )
                    sleep(self.THROTTLE_SLEEP)
                else:
                    logger.warning(f"Unknown error, retrying: {e}")
                last_exception = e

        if not response:
            raise RuntimeError(
                f"[{self.model_id}] Could not run inference: {type(last_exception)} - {str(last_exception)}"
            )

        return response

    def _prepare_payload(
        self, messages: list, inference_config: dict | None = None, system_prompt: str | None = None, **kwargs
    ):
        payload = dict(modelId=self.model_id, messages=messages, **kwargs)
        if inference_config is not None:
            payload.update({"inferenceConfig": inference_config})
        if system_prompt:
            payload.update({"system": [{"text": system_prompt}]})
        return payload

    def _process_response(self, response: dict):
        # A content-filtered invocation returns the {"response": BLOCKED}
        # sentinel from invoke() rather than a real model payload; it has no
        # "output" key, so pass it (and any other output-less shape) straight
        # through instead of KeyError-ing — the caller then sees the blocked path.
        if "output" not in response:
            return response
        return response["output"]["message"]

    def __call__(self, *args, **kwargs):
        """End-to-end model invocation."""
        payload = self._prepare_payload(*args, **kwargs)
        response = self.invoke(**payload)
        return self._process_response(response)
