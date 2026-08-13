# pii-detector

LLM-based detection of Personally Identifiable Information (PII) over
conversational text, backed by [Amazon Bedrock](https://aws.amazon.com/bedrock/).

Give it a block of text; it returns a list of detected PII spans with their
category, exact value, and character offsets. It handles long inputs by
splitting them into word-boundary chunks, recovers near-miss labels the model
emits (`DATE` → `DATES`), and retries transient Bedrock errors.

```python
[
    {"pii_entity_type": "PRIVATE_NAMES", "pii_entity_value": "John Smith", "start": 12, "end": 22},
    {"pii_entity_type": "CONTACT_INFO",  "pii_entity_value": "john.smith@example.com", "start": 74, "end": 96},
    ...
]
```

## 1. Install

Requires **Python ≥ 3.11** and AWS credentials with Bedrock access. There is no
build step — install the one dependency and run from the repo root:

```bash
cd pii-detector
python -m venv .venv && source .venv/bin/activate
pip install boto3
```

Because there's no packaging, `pii_detector` is only importable when the repo
root is on `sys.path`. Run scripts as modules from the repo root
(`python -m examples.detect`) or export `PYTHONPATH=.` before running a file
directly — being *in* the repo root is not enough, since Python puts the
script's own directory on the path, not your working directory.

### AWS credentials

The detector talks to Bedrock through boto3's **default credential chain**.
Point it at an account using the standard environment variables:

```bash
export AWS_PROFILE=my-bedrock-profile     # or set AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
export AWS_REGION=us-east-1
```

You also need [model access enabled](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
in the Bedrock console for whichever model you select.

## 2. Run

The bundled example detects PII in a sample string. Run it as a module from the
repo root so `pii_detector` resolves:

```bash
python -m examples.detect
```

Or from your own code (run from the repo root with `PYTHONPATH=.` set):

```python
from pii_detector import BedrockInferencer, PiiDetector

inferencer = BedrockInferencer(
    model_id="openai.gpt-oss-20b-1:0",
    region="us-east-1",
)
detector = PiiDetector(inferencer)

for span in detector("Call me at 555-0142 — Jane"):
    print(span)
```

`model_id` is any Bedrock Converse model id or inference-profile id — e.g.
`us.anthropic.claude-sonnet-4-5-20250929-v1:0`, `amazon.nova-lite-v1:0`,
`mistral.mistral-large-2407-v1:0`.

## 3. Code snippet examples

### Basic detection

```python
from pii_detector import BedrockInferencer, PiiDetector

detector = PiiDetector(
    BedrockInferencer(model_id="openai.gpt-oss-20b-1:0")
)
detections = detector("My SSN is 123-45-6789 and I bank with account 998877.")
# -> [{'pii_entity_type': 'IDENTIFICATION', 'pii_entity_value': '123-45-6789',
#      'start': 10, 'end': 21}, ...]
```

### Tuning sampling

Sampling knobs are per-thread and set on the inferencer before detection:

```python
inferencer = BedrockInferencer(
    model_id="openai.gpt-oss-20b-1:0", max_tokens=4096
)
inferencer.set_sampling(temperature=0.0, top_p=1.0)   # deterministic-ish
detector = PiiDetector(inferencer)
detector(text)
```

`repetition_penalty` and `seed` are forwarded via `additionalModelRequestFields`
and honoured by model families whose Converse request body accepts them (Mistral,
Qwen, OSS-GPT, DeepSeek); Claude and Nova have neither, so passing them there
raises a `ValidationException` at request time.

### Extending the category vocabulary

Inject extra PII categories, drop conflicting public-exclusion rules, and add
few-shot examples — all optional:

```python
detector = PiiDetector(
    BedrockInferencer(model_id="openai.gpt-oss-20b-1:0"),
    extra_definitions={"COMPANY_NAME": "Names of the customer's employer"},
    extra_public_drops=["Business addresses or publicly known locations"],
    extra_examples=[{
        "text": "I work at Acme Corp.",
        "pii": [{"pii_entity_type": "COMPANY_NAME", "pii_entity_value": "Acme Corp"}],
    }],
)
```

### Plugging in a custom backend

`PiiDetector` only needs an object matching the `Inferencer` protocol — a
callable taking Bedrock-Converse-shaped `messages` and returning the assistant's
text. You are not tied to Bedrock:

```python
from pii_detector import PiiDetector, Inferencer

class MyInferencer:
    model_id = "my-model"

    def __call__(self, messages: list[dict]) -> str:
        prompt = messages[0]["content"][0]["text"]
        return my_llm_call(prompt)          # must return a JSON list of detections

    def set_sampling(self, **kwargs) -> None:
        pass                                # no-op if you don't support sampling

detector = PiiDetector(MyInferencer())
```

If your backend has a context-window limit, raise
`pii_detector.ContextLengthExceeded` from `__call__` when the input overflows;
the detector will split the text and retry automatically.

## PII categories

The detector prompts for these categories out of the box (see
[`pii_detector/templates.py`](pii_detector/templates.py)):

`PRIVATE_NAMES`, `PUBLIC_NAMES`, `FULL_ADDRESSES`, `PARTIAL_ADDRESSES`,
`CONTACT_INFO`, `FINANCIAL`, `IDENTIFICATION`, `LOCATION`, `PERSONAL_NUMBERS`,
`DATES`, `DIGITAL_IDS`, `CREDENTIALS`, `AGE`, `PUBLIC_URL`, `PRIVATE_URL`.

## Layout

```
pii_detector/
  detector.py            # PiiDetector: chunking, prompting, label recovery, offsets
  bedrock_inferencer.py  # BedrockInferencer: Converse adapter (Inferencer protocol)
  bedrock.py             # BedrockConverse: thin Converse client with retry/throttle
  inferencer.py          # Inferencer protocol + ContextLengthExceeded
  templates.py           # system prompt + PII category definitions
  label_recovery.py      # re-home near-miss labels onto the prompt vocabulary
  text_chunker.py        # word-based, offset-tracked windowing
examples/
  detect.py              # runnable end-to-end example
docs/
  benchmarks.md          # benchmark tables, label mappings, extended-entity defs
```

The only runtime dependency is `boto3` (see the Install section).

## Benchmarks

[`docs/benchmarks.md`](docs/benchmarks.md) holds the full detector benchmark
table, the per-dataset label mappings, and the Extended-configuration category
definitions referenced by the accompanying blog post.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the [LICENSE](LICENSE) file.
