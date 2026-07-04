"""
TF-free drop-in replacement for textattack's UniversalSentenceEncoder.

TextFooler and CLARE both gate word substitutions on sentence-embedding
similarity, and textattack's built-in constraint (UniversalSentenceEncoder)
loads its model via tensorflow_hub. TensorFlow itself crashes on import on
this machine (macOS 26.5.1 arm64: "mutex lock failed: Invalid argument" from
libc++abi), so we swap in an equivalent constraint backed by
sentence-transformers (pure PyTorch, all-MiniLM-L6-v2) instead.
"""
from sentence_transformers import SentenceTransformer
from textattack.constraints.semantics.sentence_encoders.sentence_encoder import (
    SentenceEncoder,
)


class SBERTSentenceEncoder(SentenceEncoder):
    def __init__(self, threshold=0.8, metric="cosine", **kwargs):
        super().__init__(threshold=threshold, metric=metric, **kwargs)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, sentences):
        return self.model.encode(sentences, convert_to_numpy=True)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["model"] = None
        return state

    def __setstate__(self, state):
        self.__dict__ = state
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
