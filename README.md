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

## Comparison with other tokenizers

### Live benchmark

Input: `Перепрограммировать запрограммированный компьютер непросто`

| Tokenizer | Tokens | Output |
|-----------|--------|--------|
| **Morpheme Tokenizer** | **13** | `P:пере R:программ S:ир S:ова E:ть` · `P:за R:программ S:ир S:ова S:нн E:ый` · `R:компьютер` · `R:непросто` |
| GPT-4 (tiktoken) | 22 | `Пер` `еп` `р` `ограм` `м` `иров` `ать` `зап` `р` `ограм` `м` `иров` `анны` `й` `комп` `ью` `тер` `н` `еп` `р` `ост` `о` |
| BERT multilingual | 19 | `Пер` `##еп` `##рог` `##рам` `##мир` `##овать` `за` `##про` `##гра` `##м` `##мир` `##ован` `##ный` `комп` `##ью` `##тер` `не` `##прос` `##то` |
| razdel (word-level) | 4 | `Перепрограммировать` `запрограммированный` `компьютер` `непросто` |
| pymorphy3 (lemma) | 4 | `перепрограммировать [INFN]` `запрограммировать [PRTF]` `компьютер [NOUN]` `непросто [ADVB]` |

Key observation: GPT-4 splits `программ` into 4 fragments (`р` + `ограм` + `м` + `иров`) — destroying the root. Morpheme Tokenizer keeps `R:программ` intact across both words, enabling **embedding sharing** between related forms.

### Feature comparison

| | **Morpheme Tokenizer** | **tiktoken (GPT-4)** | **SentencePiece** | **BERT WordPiece** | **pymorphy3** | **Mystem** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Approach** | Dictionary + cascade | BPE (statistical) | BPE / Unigram | WordPiece (statistical) | Grammar rules | Statistical + rules |
| **Morpheme segmentation** | ✅ root/prefix/suffix/ending | ❌ | ❌ | ❌ | ❌ | ⚠️ stem + flex |
| **Morpheme type labels** | ✅ `R:` `P:` `S:` `E:` | ❌ | ❌ | ❌ | POS tags only | POS tags only |
| **Root sharing** | ✅ same root → same token | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Russian dictionary** | 100K (Tikhonov) | — | — | — | 400K (OpenCorpora) | ~300K |
| **Lemmatization** | ✅ via pymorphy3 | ❌ | ❌ | ❌ | ✅ native | ✅ native |
| **Typo tolerance** | ✅ edit distance 1 | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Multilingual** | ❌ Russian + Latin fallback | ✅ | ✅ | ✅ | ❌ Russian | ❌ Russian |
| **Vocab size** | 32K | 100K | 32–64K | 30–120K | — | — |
| **Speed** | ~50K tok/s | ~1M tok/s | ~1M tok/s | ~500K tok/s | ~100K tok/s | ~500K tok/s |
| **License** | MIT | MIT | Apache-2.0 | Apache-2.0 | MIT | Proprietary |

### Why morpheme-level matters

```
BPE (statistical splits):           Morpheme Tokenizer (linguistic splits):
програм|миро|вание                   R:программ  S:ир S:ова S:ни E:е
програм|мист                         R:программ  S:ист
програм|мы                           R:программ  E:ы
пере|програм|миро|вать               P:пере R:программ  S:ир S:ова E:ть
                                     ↑ same root token = shared embedding
```

**Benefits for downstream models:**
- **Embedding sharing** — words with the same root share the `R:программ` embedding
- **Morphological awareness** — model sees `S:ист` = "person who does" across all professions
- **Grammar/semantics separation** — `E:ы` (plural ending) gets its own embedding, separate from meaning
- **Compact vocabulary** — 32K tokens cover Russian better than 100K BPE tokens, because morphemes are reusable

**Trade-offs:**
- Slower than pure BPE (~50K vs ~1M tok/s) — dictionary lookup + pymorphy3
- Russian-only (Latin text falls back to character-level)
- 100K dictionary ≈ 70-80% coverage (vs BPE's 100% statistical coverage)

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
