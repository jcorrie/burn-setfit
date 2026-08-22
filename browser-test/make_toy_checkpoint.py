#!/usr/bin/env python3
"""Generate a toy HuggingFace-format MiniLM checkpoint.

The browser harness needs the same three files a real checkpoint has --
`config.json`, `model.safetensors`, `tokenizer.json` -- but nothing about the
runtime questions it answers (threads, entropy, event-loop yielding, memory)
depends on the weights being any good. A 32-wide, 2-layer body with a 30-token
vocabulary exercises every code path a real MiniLM does, in 200 KB instead of
90 MB, and needs no network.

Names and layout follow HuggingFace BERT exactly, because that is what
`Checkpoint::from_files` assumes: PyTorch's `[out, in]` Linear layout, and
`LayerNorm.weight`/`.bias` rather than Burn's `gamma`/`beta`. The loader's key
remap and transpose are therefore exercised too.
"""

import json
import random
import struct
import sys
from pathlib import Path

HIDDEN = 32
HEADS = 2
LAYERS = 2
INTERMEDIATE = 64
VOCAB = 64
MAX_POS = 512
TYPE_VOCAB = 2
SEED = 20260822


def tensors():
    """HF tensor name -> shape, in HuggingFace BERT naming and layout."""
    t = {
        "embeddings.word_embeddings.weight": [VOCAB, HIDDEN],
        "embeddings.position_embeddings.weight": [MAX_POS, HIDDEN],
        "embeddings.token_type_embeddings.weight": [TYPE_VOCAB, HIDDEN],
        "embeddings.LayerNorm.weight": [HIDDEN],
        "embeddings.LayerNorm.bias": [HIDDEN],
    }
    for i in range(LAYERS):
        p = f"encoder.layer.{i}"
        for proj in ("query", "key", "value"):
            t[f"{p}.attention.self.{proj}.weight"] = [HIDDEN, HIDDEN]
            t[f"{p}.attention.self.{proj}.bias"] = [HIDDEN]
        t[f"{p}.attention.output.dense.weight"] = [HIDDEN, HIDDEN]
        t[f"{p}.attention.output.dense.bias"] = [HIDDEN]
        t[f"{p}.attention.output.LayerNorm.weight"] = [HIDDEN]
        t[f"{p}.attention.output.LayerNorm.bias"] = [HIDDEN]
        # PyTorch Linear is [out, in]; Burn is [in, out]. The adapter transposes.
        t[f"{p}.intermediate.dense.weight"] = [INTERMEDIATE, HIDDEN]
        t[f"{p}.intermediate.dense.bias"] = [INTERMEDIATE]
        t[f"{p}.output.dense.weight"] = [HIDDEN, INTERMEDIATE]
        t[f"{p}.output.dense.bias"] = [HIDDEN]
        t[f"{p}.output.LayerNorm.weight"] = [HIDDEN]
        t[f"{p}.output.LayerNorm.bias"] = [HIDDEN]
    return t


def values(name, count, rng):
    """LayerNorm scales start at 1 and biases at 0, as in a real checkpoint;
    everything else is small random, so a forward pass stays finite."""
    if name.endswith("LayerNorm.weight"):
        return [1.0] * count
    if name.endswith(".bias"):
        return [0.0] * count
    return [rng.gauss(0.0, 0.02) for _ in range(count)]


def write_safetensors(path):
    rng = random.Random(SEED)
    header, blobs, offset = {}, [], 0
    for name, shape in tensors().items():
        count = 1
        for d in shape:
            count *= d
        blob = struct.pack(f"<{count}f", *values(name, count, rng))
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, offset + len(blob)],
        }
        blobs.append(blob)
        offset += len(blob)

    header_bytes = json.dumps(header, separators=(",", ":")).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for blob in blobs:
            f.write(blob)


def write_config(path):
    path.write_text(
        json.dumps(
            {
                "architectures": ["BertModel"],
                "model_type": "bert",
                "hidden_size": HIDDEN,
                "num_attention_heads": HEADS,
                "num_hidden_layers": LAYERS,
                "intermediate_size": INTERMEDIATE,
                "vocab_size": VOCAB,
                "max_position_embeddings": MAX_POS,
                "type_vocab_size": TYPE_VOCAB,
                "hidden_dropout_prob": 0.0,
                "layer_norm_eps": 1e-12,
            },
            indent=2,
        )
    )


def write_tokenizer(path):
    """The same a-z WordPiece vocabulary `tests/common/mod.rs` uses, so token
    counts stay trivially predictable and the browser and native harnesses
    agree on what a chunk is."""
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
    for i, c in enumerate("abcdefghijklmnopqrstuvwxyz"):
        vocab[c] = i + 4
    for i, c in enumerate([".", "!", "?", ","]):
        vocab[c] = i + 30

    path.write_text(
        json.dumps(
            {
                "version": "1.0",
                "truncation": None,
                # Deliberately padding to a fixed length: a published
                # `tokenizer.json` does this (all-MiniLM-L6-v2 pads to 128) and
                # `Tokenizer::from_bytes` must strip it. If the browser build
                # ever stops stripping it, this file will catch it.
                "padding": {
                    "strategy": {"Fixed": 128},
                    "direction": "Right",
                    "pad_to_multiple_of": None,
                    "pad_id": 0,
                    "pad_type_id": 0,
                    "pad_token": "[PAD]",
                },
                "added_tokens": [],
                "normalizer": {
                    "type": "BertNormalizer",
                    "clean_text": True,
                    "handle_chinese_chars": True,
                    "strip_accents": None,
                    "lowercase": True,
                },
                "pre_tokenizer": {"type": "BertPreTokenizer"},
                "post_processor": None,
                "decoder": None,
                "model": {
                    "type": "WordPiece",
                    "unk_token": "[UNK]",
                    "continuing_subword_prefix": "##",
                    "max_input_chars_per_word": 100,
                    "vocab": vocab,
                },
            },
            indent=2,
        )
    )


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "public/checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    write_config(out / "config.json")
    write_safetensors(out / "model.safetensors")
    write_tokenizer(out / "tokenizer.json")
    for f in sorted(out.iterdir()):
        print(f"{f}  {f.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
