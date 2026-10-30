"""Lossless UTF-8 **kern serialization; no learned vocabulary or voice limit."""
import hashlib
import json
import re
from collections import Counter


class KernTokenizer:
    PAD, BOS, EOS = 0, 1, 2
    vocab_size = 259
    version = "kern-utf8-v1"

    def metadata(self):
        spec = {"version": self.version, "encoding": "utf-8", "byte_offset": 3,
                "pad": self.PAD, "bos": self.BOS, "eos": self.EOS,
                "vocab_size": self.vocab_size}
        spec["sha256"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        return spec

    def encode(self, text):
        # Normalize line endings only. Tabs, chord spaces and all metadata survive.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        return [b + 3 for b in text.encode("utf-8")] + [self.EOS]

    def token_bytes(self, token):
        if not 3 <= token < self.vocab_size:
            raise ValueError(f"Invalid body token {token}")
        return bytes([token - 3])

    def decode(self, ids, require_eos=True):
        ids = [int(i) for i in ids]
        if ids and ids[0] == self.BOS:
            ids = ids[1:]
        payload = []
        for i, token in enumerate(ids):
            if token == self.EOS:
                if any(t != self.PAD for t in ids[i + 1:]):
                    raise ValueError("Non-padding tokens after EOS")
                return bytes(payload).decode("utf-8", errors="strict")
            if not 3 <= token < self.vocab_size:
                raise ValueError(f"Invalid body token {token}")
            payload.extend(self.token_bytes(token))
        if require_eos:
            raise ValueError("Missing EOS (generation reached its token limit)")
        return bytes(payload).decode("utf-8", errors="strict")


class KernFieldTokenizer(KernTokenizer):
    """Frequent complete Kern fields + byte fallback; separators remain explicit.

    Vocabulary is fitted on TRAIN only. Unknown notes/metadata remain lossless.
    This reduces decoding steps without imposing pitch, rhythm or voice limits.
    """
    version = "kern-fields-v2"

    def __init__(self, pieces):
        if len(set(pieces)) != len(pieces) or any(not p or re.search(r"[\s]", p) for p in pieces):
            raise ValueError("Vocabulary fields must be unique and contain no whitespace")
        self.pieces = list(pieces)
        self.lookup = {p: 259 + i for i, p in enumerate(pieces)}
        self.vocab_size = 259 + len(pieces)

    @classmethod
    def fit(cls, texts, max_vocab=2048, min_frequency=2):
        counts = Counter()
        for text in texts:
            counts.update(re.findall(r"[^\s]+", text))
        # Rank by bytes saved, deterministically; don't spend IDs on 1-byte fields.
        candidates = sorted((p for p, c in counts.items() if c >= min_frequency and len(p.encode()) > 1),
                            key=lambda p: (-counts[p] * (len(p.encode()) - 1), p))
        return cls(candidates[:max(0, max_vocab - 259)])

    def metadata(self):
        spec = {"version": self.version, "pieces": self.pieces, "pad": 0, "bos": 1, "eos": 2,
                "byte_offset": 3, "encoding": "utf-8", "vocab_size": self.vocab_size}
        spec["sha256"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
        return spec

    def encode(self, text):
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        result = []
        for field in re.findall(r"[^\s]+|\s", text):
            if field in self.lookup:
                result.append(self.lookup[field])
            else:
                result.extend(b + 3 for b in field.encode("utf-8"))
        return result + [self.EOS]

    def token_bytes(self, token):
        if token < 259:
            return super().token_bytes(token)
        if token >= self.vocab_size:
            raise ValueError(f"Invalid field token {token}")
        return self.pieces[token - 259].encode("utf-8")


def tokenizer_from_metadata(spec=None):
    if spec is None:
        return KernTokenizer()
    if spec.get("version") == KernFieldTokenizer.version:
        tok = KernFieldTokenizer(spec["pieces"])
    elif spec.get("version") == KernTokenizer.version:
        tok = KernTokenizer()
    else:
        raise ValueError("Unsupported tokenizer version")
    if tok.metadata() != spec:
        raise ValueError("Tokenizer vocabulary/hash mismatch")
    return tok
