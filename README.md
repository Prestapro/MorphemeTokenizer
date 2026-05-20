# Morpheme Tokenizer

Root-aware morpheme tokenizer for Russian text. Uses [Tikhonov's morpheme dictionary](https://ru.wikipedia.org/wiki/Тихонов,_Александр_Николаевич_(лингвист)) (100K+ words) for linguistically accurate decomposition, with character-level fallback for unknown words.

**[Live Demo →](https://prestapro.github.io/MorphemeTokenizer/)**

## What it does

Unlike BPE/SentencePiece which splits words statistically, this tokenizer splits words **by morphemes** — root, prefix, suffix, ending:

```
программирование → P:пере R:программ S:ир S:ова S:ни E:е
перепрограммировать → P:пере R:программ S:ир S:ова E:ть
```

Every token carries its **morpheme type**, so downstream models can distinguish semantic roots from grammar markers.

## Token types

| Type | Prefix | Example | Meaning |
|------|--------|---------|---------|
| Root | `R:` | `R:программ` | Semantic core |
| Prefix | `P:` | `P:пере` | Derivational prefix |
| Suffix | `S:` | `S:ирова` | Derivational suffix |
| Ending | `E:` | `E:ние` | Inflectional ending |
| BPE | `B:` | `B:P` | Character fallback |
| Punct | `X:` | `X:.` | Punctuation |
| Number | `N:` | `N:4 N:2` | Digit-by-digit |

## Quick start

```python
from morpheme_tokenizer import MorphemeTokenizer

tok = MorphemeTokenizer.from_tikhonov("data/tikhonov_morphemes.json", char_fallback=True)

tokens = tok.tokenize("Нейронные сети обучаются")
# → ['R:нейронны', 'E:е', '<sep>', 'R:сет', 'E:и', '<sep>', 'R:обуча', 'E:ются']

# Encode to IDs for model training
ids = tok.encode("Нейронные сети")
# → [1, 4523, 127, 4, 8901, 98, 2]  (with BOS/EOS)

# Decode back
text = tok.decode(ids)
```

## Lookup cascade

When a word isn't found directly in the dictionary, the tokenizer tries 7 fallback strategies:

1. **Exact match** in Tikhonov dictionary
2. **pymorphy3 lemma** → lookup the lemma
3. **Strip -ся/-сь** → retry (reflexive verbs)
4. **Aspect pairs** — обучать→обучить
5. **Derivational forms** — verb→noun, adj→noun
6. **Prefix stripping** — обучать→учить→root `уч`
7. **Fuzzy match** — edit distance 1 (typo tolerance)

## Dictionary

| File | Size | Contents |
|------|------|----------|
| `data/tikhonov_morphemes.json` | 25 MB | 100,097 words — roots, prefixes, suffixes, endings |
| `data/morpheme_extensions.json` | 2.2 MB | Auto-generated extensions from corpus analysis |

### Coverage

| Metric | Value |
|--------|-------|
| Dictionary words | 100,097 |
| Unique roots | 15,389 |
| Unique suffixes | 8,315 |
| Unique prefixes | 65 |
| Vocab size (with char fallback) | 32,236 |
| Typical text coverage | 70–80% (rest → char fallback) |

## Dependencies

```
pymorphy3    # optional: enables lemma lookup (levels 2-6)
```

Without pymorphy3, only exact dictionary match works (level 1).

## Files

```
├── index.html                  # Interactive web demo (standalone, no backend)
├── morpheme_tokenizer.py       # Python tokenizer
├── data/
│   ├── tikhonov_morphemes.json # Tikhonov morpheme dictionary (100K words)
│   └── morpheme_extensions.json # Corpus-derived extensions
└── README.md
```

## Part of Logos

This tokenizer is a component of [Logos](https://github.com/Prestapro/logos) — a symbolic AI engine for Russian NLU. In the training pipeline, morpheme tokens feed a 7-channel semantic encoder where each token carries: token ID, type ID, position, word length, semantic type, concept ID, and KG memory signal.

## License

MIT
