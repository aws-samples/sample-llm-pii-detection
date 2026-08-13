"""Minimal end-to-end PII detection examples.

Run from the repo root as a module (so `pii_detector` is importable):
    python -m examples.detect

Requires AWS credentials with Bedrock access on the default credential chain
(e.g. `export AWS_PROFILE=my-profile`) and access to the chosen model.
"""
import sys

from pii_detector import BedrockInferencer, PiiDetector

MODEL_ID = "openai.gpt-oss-20b-1:0"
REGION = "us-east-1"

_CREDS_HELP = f"""\
Could not reach Bedrock. Check that:
  - AWS credentials are set (export AWS_PROFILE=... or AWS_ACCESS_KEY_ID /
    AWS_SECRET_ACCESS_KEY), and are not expired.
  - The credentials have Bedrock InvokeModel permission in {REGION}.
  - Model access for '{MODEL_ID}' is enabled in the Bedrock console for {REGION}.

See https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html
Original error: {{err}}"""


def _preflight() -> None:
    """Fail fast with actionable guidance if Bedrock is unreachable.

    PiiDetector swallows inference errors (it logs and returns an empty list),
    so a bad-credentials run would otherwise look like "no PII found". This
    makes one direct inferencer call — bypassing the detector's retry/swallow
    loop — so credential and model-access failures surface clearly up front.
    """
    try:
        # max_tokens=1: we only care whether the call succeeds, not its output.
        BedrockInferencer(model_id=MODEL_ID, region=REGION, max_tokens=1)(
            [{"role": "user", "content": [{"text": "ping"}]}]
        )
    except Exception as err:
        print(_CREDS_HELP.format(err=err), file=sys.stderr)
        sys.exit(1)

TEXT = (
    "Hi, this is John Smith. My account number is 4455-2211 and you can reach "
    "me at john.smith@example.com or on 555-0142. I live at 82 Oak Street, "
    "Springfield, 02412."
)


def _print(detections: list) -> None:
    for d in detections:
        print(f"{d['pii_entity_type']:<18} [{d['start']:>4}:{d['end']:<4}]  {d['pii_entity_value']!r}")


def basic() -> None:
    """Detection with the built-in PII category vocabulary."""
    inferencer = BedrockInferencer(model_id=MODEL_ID, region=REGION)
    # `seed` is honoured by gpt-oss (the default MODEL_ID); Claude/Nova would
    # reject it with a ValidationException.
    inferencer.set_sampling(temperature=0.9, seed=42)
    detector = PiiDetector(inferencer)
    _print(detector(TEXT))


def with_extra_definitions() -> None:
    """Extend the vocabulary with a custom category.

    This is the ``COMPANY_NAME`` set used for the public benchmark datasets
    (Gretel / Nemotron). The base prompt tells the model NOT to flag business
    information, so injecting a company category also requires dropping that
    conflicting public-exclusion line — otherwise the model keeps ignoring it.
    A single few-shot example anchors the new category.
    """
    extra_definitions = {
        "COMPANY_NAME": (
            "Any company, employer, brand, or organization name that appears in "
            "the document (e.g., \"Acme Corp\", \"Goldman Sachs\", \"BreezyGoods\"). "
            "Flag every occurrence even when the company is mentioned multiple "
            "times. This includes companies named in headers, contract parties, "
            "recall notices, or product references — flag them regardless of "
            "whether a specific person is tied to the company."
        ),
    }
    extra_public_drops = [
        "Business addresses or publicly known locations",  # conflicts with COMPANY_NAME
    ]
    extra_examples = [
        {
            "text": (
                "**Vehicle Recall Notice** for AutoPioneers XR3. Manufacturer: "
                "AutoPioneers. Owners may contact AutoPioneers customer service."
            ),
            "pii": [
                {"pii_entity_type": "COMPANY_NAME", "pii_entity_value": "AutoPioneers"},
                {"pii_entity_type": "COMPANY_NAME", "pii_entity_value": "AutoPioneers"},
                {"pii_entity_type": "COMPANY_NAME", "pii_entity_value": "AutoPioneers"},
            ],
        },
    ]

    detector = PiiDetector(
        BedrockInferencer(model_id=MODEL_ID, region=REGION),
        extra_definitions=extra_definitions,
        extra_public_drops=extra_public_drops,
        extra_examples=extra_examples,
    )
    _print(detector("Miguel Richardson works at BreezyGoods and banks with Goldman Sachs."))


if __name__ == "__main__":
    _preflight()
    print("== basic ==")
    basic()
    print("\n== with extra definitions (COMPANY_NAME) ==")
    with_extra_definitions()
