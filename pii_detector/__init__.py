"""PII detection over conversational text via a Bedrock-backed LLM.

Public API:
    PiiDetector          — the detector (chunking, prompting, label recovery).
    BedrockInferencer    — Bedrock Converse adapter satisfying the Inferencer protocol.
    BedrockConverse      — thin Bedrock Converse client (used by BedrockInferencer).
    Inferencer           — protocol for plugging in a custom backend.
    ContextLengthExceeded — signal an Inferencer raises on context overflow.
"""
from pii_detector.detector import PiiDetector
from pii_detector.bedrock_inferencer import BedrockInferencer
from pii_detector.bedrock import BedrockConverse
from pii_detector.inferencer import Inferencer, ContextLengthExceeded

__all__ = [
    "PiiDetector",
    "BedrockInferencer",
    "BedrockConverse",
    "Inferencer",
    "ContextLengthExceeded",
]
