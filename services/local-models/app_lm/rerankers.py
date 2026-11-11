"""Reranker adapters for the local-models service.

Two adapters (stage-0 S7 evidence):
- QwenYesNoReranker — the OFFICIAL Qwen3-Reranker recipe: chat template with a
  fixed instruction, last-position logits of the "yes"/"no" tokens, log-softmax
  over the pair -> P(yes). Implemented via transformers AutoModelForCausalLM, so
  it works under the pinned transformers 4.57.6 regardless of the
  sentence-transformers version. S7 proved this is bit-identical to the ST 6.x
  CrossEncoder path (sigmoid(raw) == P(yes), abs diff 0.0 at 6 d.p.).
- CrossEncoderReranker — classic sequence-classification cross-encoders
  (BAAI/bge-reranker-v2-m3) via sentence-transformers CrossEncoder.

Scores are raw model outputs; scales are NOT comparable across models and a 0.5
threshold is meaningless without calibration (ADR 002).
"""

from __future__ import annotations

from typing import Any, Protocol

RERANK_INSTRUCTION_V1 = (
    "Given a question about a contract, judge whether the document fragment "
    "contains information relevant to answering it"
)

_QWEN_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based "
    'on the Query and the Instruct provided. Note that the answer can only be '
    '"yes" or "no".<|im_end|>\n<|im_start|>user\n'
)
_QWEN_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
_QWEN_MAX_LENGTH = 8192


class RerankerAdapter(Protocol):
    model_id: str
    revision: str
    adapter_id: str

    def score(self, query: str, documents: list[str]) -> list[float]: ...


class QwenYesNoReranker:
    adapter_id = "qwen_yes_no_v1"

    def __init__(self, model_id: str, revision: str, *, batch_size: int = 4) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.revision = revision
        self._batch_size = batch_size
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, revision=revision, padding_side="left"
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=torch.float32
        ).eval()
        self._token_true = self._tokenizer.convert_tokens_to_ids("yes")
        self._token_false = self._tokenizer.convert_tokens_to_ids("no")
        self._prefix = self._tokenizer.encode(_QWEN_PREFIX, add_special_tokens=False)
        self._suffix = self._tokenizer.encode(_QWEN_SUFFIX, add_special_tokens=False)

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        torch = self._torch
        texts = [
            f"<Instruct>: {RERANK_INSTRUCTION_V1}\n<Query>: {query}\n<Document>: {doc}"
            for doc in documents
        ]
        inputs: Any = self._tokenizer(
            texts, padding=False, truncation="longest_first",
            return_attention_mask=False,
            max_length=_QWEN_MAX_LENGTH - len(self._prefix) - len(self._suffix),
        )
        for i, ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self._prefix + ids + self._suffix
        inputs = self._tokenizer.pad(
            inputs, padding=True, return_tensors="pt", max_length=_QWEN_MAX_LENGTH
        )
        with torch.no_grad():
            logits_last = self._model(**inputs).logits[:, -1, :]
        stacked = torch.stack(
            [logits_last[:, self._token_false], logits_last[:, self._token_true]], dim=1
        )
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)
        return [float(v) for v in log_probs[:, 1].exp()]

    def score(self, query: str, documents: list[str]) -> list[float]:
        scores: list[float] = []
        for start in range(0, len(documents), self._batch_size):
            scores.extend(self._score_batch(query, documents[start : start + self._batch_size]))
        return scores


class CrossEncoderReranker:
    adapter_id = "cross_encoder_v1"

    def __init__(self, model_id: str, revision: str, *, batch_size: int = 8) -> None:
        from sentence_transformers import CrossEncoder

        self.model_id = model_id
        self.revision = revision
        self._batch_size = batch_size
        self._model = CrossEncoder(model_id, revision=revision, device="cpu")

    def score(self, query: str, documents: list[str]) -> list[float]:
        pairs = [(query, doc) for doc in documents]
        return [float(s) for s in self._model.predict(pairs, batch_size=self._batch_size)]


def load_reranker(
    adapter: str, model_id: str, revision: str, *, batch_size: int
) -> RerankerAdapter:
    if adapter == "qwen_yes_no":
        return QwenYesNoReranker(model_id, revision, batch_size=batch_size)
    if adapter == "cross_encoder":
        return CrossEncoderReranker(model_id, revision, batch_size=batch_size)
    raise ValueError(f"unknown reranker adapter '{adapter}'")
