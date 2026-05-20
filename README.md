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

### Three levels of tokenization

Tokenizers work at different granularity levels. Most tools operate at only one level:

```
Level 1 — Word:     "Перепрограммировать" → 1 token (whole word)
                    NLTK, SpaCy, razdel, Gensim

Level 2 — Subword:  "Перепрограммировать" → 8 tokens (statistical fragments)
                    tiktoken (GPT-4), SentencePiece, BERT WordPiece

Level 3 — Morpheme: "Перепрограммировать" → 5 tokens (linguistic morphemes)
                    Morpheme Tokenizer  ← you are here
```

Word-level tokenizers (NLTK, SpaCy) don't decompose words at all — vocabulary explodes with inflected forms. Subword tokenizers (BPE) decompose statistically — fragments have no linguistic meaning. **Morpheme Tokenizer is the only tool that decomposes into linguistically meaningful units with type labels.**

### Live benchmark

Input: `Перепрограммировать запрограммированный компьютер непросто`

| Tokenizer | Level | Tokens | Output |
|-----------|-------|--------|--------|
| **Morpheme Tokenizer** | Morpheme | **13** | `P:пере R:программ S:ир S:ова E:ть` · `P:за R:программ S:ир S:ова S:нн E:ый` · `R:компьютер` · `R:непросто` |
| GPT-4 (tiktoken) | Subword | 22 | `Пер` `еп` `р` `ограм` `м` `иров` `ать` `зап` `р` `ограм` `м` `иров` `анны` `й` `комп` `ью` `тер` `н` `еп` `р` `ост` `о` |
| BERT multilingual | Subword | 19 | `Пер` `##еп` `##рог` `##рам` `##мир` `##овать` `за` `##про` `##гра` `##м` `##мир` `##ован` `##ный` `комп` `##ью` `##тер` `не` `##прос` `##то` |
| SpaCy (`ru_core_news_sm`) | Word | 4 | `Перепрограммировать` `запрограммированный` `компьютер` `непросто` |
| NLTK (TreebankWord) | Word | 4 | `Перепрограммировать` `запрограммированный` `компьютер` `непросто` |
| razdel | Word | 4 | `Перепрограммировать` `запрограммированный` `компьютер` `непросто` |
| pymorphy3 | Lemma | 4 | `перепрограммировать [INFN]` `запрограммировать [PRTF]` `компьютер [NOUN]` `непросто [ADVB]` |
| Gensim (Phrases) | Word+bigrams | 4 | Same as word-level (no bigrams in this input) |

Key observation: GPT-4 splits `программ` into 4 fragments (`р` + `ограм` + `м` + `иров`) — destroying the root. Word-level tools keep the whole word but can't decompose it. **Only Morpheme Tokenizer preserves the root `R:программ` as a reusable semantic unit.**

### Feature comparison

| | **Morpheme Tokenizer** | **tiktoken (GPT-4)** | **SentencePiece** | **BERT WordPiece** | **SpaCy** | **NLTK** | **pymorphy3** | **Mystem** |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Level** | Morpheme | Subword | Subword | Subword | Word | Word | Lemma | Lemma |
| **Approach** | Dictionary + cascade | BPE (statistical) | BPE / Unigram | WordPiece | Rules + model | Regex / rules | Grammar rules | Statistical + rules |
| **Morpheme segmentation** | ✅ root/prefix/suffix/ending | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ⚠️ stem + flex |
| **Morpheme type labels** | ✅ `R:` `P:` `S:` `E:` | ❌ | ❌ | ❌ | ❌ | ❌ | POS tags only | POS tags only |
| **Root sharing** | ✅ same root → same token | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Handles OOV words** | ✅ char fallback | ✅ BPE | ✅ BPE | ✅ `[UNK]` | ✅ | ✅ | ✅ | ✅ |
| **POS tagging** | ⚠️ via pymorphy3 | ❌ | ❌ | ❌ | ✅ native | ❌ | ✅ native | ✅ native |
| **NER** | ❌ | ❌ | ❌ | ❌ | ✅ native | ❌ | ❌ | ❌ |
| **Dependency parsing** | ❌ | ❌ | ❌ | ❌ | ✅ native | ❌ | ❌ | ❌ |
| **Bigram/collocation** | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Russian dictionary** | 100K (Tikhonov) | — | — | — | model-based | — | 400K (OpenCorpora) | ~300K |
| **Lemmatization** | ✅ via pymorphy3 | ❌ | ❌ | ❌ | ✅ | ❌ | ✅ native | ✅ native |
| **Typo tolerance** | ✅ edit distance 1 | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Multilingual** | ❌ Russian + Latin fallback | ✅ | ✅ | ✅ | ✅ (75+ langs) | ✅ | ❌ Russian | ❌ Russian |
| **Vocab size** | 32K | 100K | 32–64K | 30–120K | word-level | word-level | — | — |
| **Speed** | ~50K tok/s | ~1M tok/s | ~1M tok/s | ~500K tok/s | ~100K tok/s | ~500K tok/s | ~100K tok/s | ~500K tok/s |
| **License** | MIT | MIT | Apache-2.0 | Apache-2.0 | MIT | Apache-2.0 | MIT | Proprietary |

### Why morpheme-level matters

#### 1. Root sharing across word family

GPT-4 uses **16 different fragments** for 7 words from the same family. Morpheme Tokenizer uses **1 root token**:

```
Word                          Morpheme Tokenizer              GPT-4 (tiktoken)
──────────────────────────────────────────────────────────────────────────────
программа                     R:программ  E:а                 п|р|ограм|ма
программист                   R:программ  S:ист               п|р|ограм|м|ист
программировать               R:программ  S:ир S:ова E:ть     п|р|ограм|м|иров|ать
программирование              R:программ  S:ир S:ова S:ни E:е п|р|ограм|м|иров|ание
программный                   R:программ  S:н E:ый            п|р|ограм|м|ный
перепрограммировать           P:пере R:программ S:ир S:ова E:ть п|ер|еп|р|ограм|м|иров|ать
запрограммированный           P:за R:программ S:ир S:ова S:нн E:ый з|ап|р|ограм|м|иров|анны|й
                              ↑ 1 shared root                  ↑ 16 unrelated fragments
```

#### 2. Rare word generalization

Model has seen `программа` 1000 times but `перепрограммировать` only once:

| | GPT-4 | Morpheme Tokenizer |
|---|---|---|
| `перепрограммировать` embedding | ❌ Barely trained (1 occurrence) | ✅ `R:программ` already trained from 1000 hits |
| Semantic link to `программа` | ❌ None — different subword fragments | ✅ Same root token = shared embedding |

#### 3. 3× smaller vocabulary, better coverage

| | Morpheme | GPT-4 |
|---|---|---|
| Vocab size | **32K** | 100K |
| «программ» family | **1 token** for 7 words | 16 fragments |
| Embedding matrix (dim 1024) | 33M params | 102M params |
| Freed parameters | **+70M** for attention/FFN | — |

15,389 roots × 65 prefixes × 8,315 suffixes × 33 endings = combinatorial coverage of **millions of word forms** from 32K tokens.

#### 4. Typed morphemes = structural knowledge

```
S:ист  = "person who does" → программист, журналист, машинист
S:ни   = "process"         → программирование, обучение, чтение
E:ы    = "plural"          → программы, модели, сети
P:пере = "re-/again"       → перепрограммировать, перенастроить
```

BPE treats `ист` and `ание` as opaque character sequences. Morpheme Tokenizer gives the model **explicit morphological structure**.

#### Trade-offs

- **Speed**: ~50K vs ~1M tok/s — but tokenization is one-time preprocessing, not inference bottleneck
- **Language**: Russian only (Latin → character fallback)
- **Coverage**: 100K dictionary ≈ 70-80% of text (rest → char fallback). BPE covers 100% statistically


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
