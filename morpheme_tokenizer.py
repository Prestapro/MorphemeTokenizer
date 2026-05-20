#!/usr/bin/env python3
"""Root-aware morpheme tokenizer for Russian text.

Uses Tikhonov's morpheme dictionary for decomposition,
with BPE fallback for out-of-vocabulary words.

Architecture:
    Input text → Word tokenization → For each word:
        1. Exact lookup in Tikhonov dictionary → morphemes
        2. pymorphy3 lemma → lookup lemma in dictionary → adapt morphemes
        3. BPE fallback for unknown words

Token types:
    [ROOT:программ]  — semantic core
    [SUF:ирова]      — derivational suffix
    [END:ние]        — inflectional ending
    [PRE:пере]       — prefix
    [BPE:xyz]        — fallback subword
    [PUNCT:.]        — punctuation
    [NUM:42]         — number
    [SPACE]          — word boundary

Usage:
    from engine.training.morpheme_tokenizer import MorphemeTokenizer
    tok = MorphemeTokenizer.from_tikhonov("data/tikhonov_morphemes.json")
    tokens = tok.tokenize("программирование на Python")
    # → ['ROOT:программ', 'SUF:ир', 'SUF:ова', 'SUF:ни', 'END:е',
    #    'ROOT:на', 'BPE:Python']
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


class MorphemeTokenizer:
    """Root-aware morpheme tokenizer using Tikhonov dictionary."""
    
    # Special token IDs
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    SEP = 4
    
    # Morpheme type prefixes for tokens
    PREFIX_ROOT = "R:"
    PREFIX_SUF = "S:"
    PREFIX_END = "E:"
    PREFIX_PRE = "P:"
    PREFIX_BPE = "B:"
    PREFIX_PUNCT = "X:"
    PREFIX_NUM = "N:"
    
    def __init__(
        self,
        dictionary: dict,
        roots_index: dict,
        bpe_tokenizer=None,
        char_fallback: bool = False,
    ):
        self.dictionary = dictionary  # word -> morpheme entry
        self.roots_index = roots_index  # root -> [words]
        self.bpe_tokenizer = bpe_tokenizer
        self.char_fallback = char_fallback
        
        # Try to load pymorphy3 for lemmatization fallback
        try:
            import pymorphy3
            self._morph = pymorphy3.MorphAnalyzer()
        except ImportError:
            self._morph = None
        
        # Build vocabulary
        self._build_vocab()
    
    @staticmethod
    def _clean_morpheme(s: str) -> str:
        """Strip Tikhonov annotations from morpheme string.
        
        Examples:
            'о1, эт'      → 'о'    (homonym index + cross-ref)
            'ого, местоим.' → 'ого' (POS annotation)
            'а (пенька)'  → 'а'    (semantic gloss)
            'ий (от Абруцци)' → 'ий'
            'ен кр. ф. прич.' → 'ен'
        """
        import re
        # Strip parenthetical annotations
        s = re.sub(r'\s*\(.*?\)', '', s)
        # Strip everything after first comma or semicolon
        for sep in (',', ';'):
            if sep in s:
                s = s.split(sep)[0]
        # Strip abbreviation-style annotations (contain periods)
        # e.g. "кр. ф." "местоим." "нареч."
        if '.' in s:
            # Keep only the part before the first word with a period
            parts = s.split()
            cleaned_parts = []
            for p in parts:
                if '.' in p:
                    break
                cleaned_parts.append(p)
            s = ' '.join(cleaned_parts) if cleaned_parts else s.split('.')[0]
        # Strip trailing numeric homonym indices
        s = re.sub(r'\d+$', '', s)
        return s.strip()
    
    def _build_vocab(self):
        """Build token vocabulary from morpheme dictionary."""
        self.token2id = {
            "<pad>": self.PAD,
            "<bos>": self.BOS,
            "<eos>": self.EOS,
            "<unk>": self.UNK,
            "<sep>": self.SEP,
        }
        
        # Collect all unique morphemes by type
        roots = set()
        suffixes = set()
        endings = set()
        prefixes = set()
        
        for word, entry in self.dictionary.items():
            if entry.get("root"):
                roots.add(self._clean_morpheme(entry["root"]))
            for s in entry.get("suffixes", []):
                cleaned = self._clean_morpheme(s)
                if cleaned:
                    suffixes.add(cleaned)
            if entry.get("ending"):
                endings.add(self._clean_morpheme(entry["ending"]))
            for p in entry.get("prefixes", []):
                cleaned = self._clean_morpheme(p)
                if cleaned:
                    prefixes.add(cleaned)
        
        # Assign IDs: roots first (most important), then affixes
        idx = 5  # Start after special tokens
        
        # Roots
        for r in sorted(roots):
            token = f"{self.PREFIX_ROOT}{r}"
            self.token2id[token] = idx
            idx += 1
        
        # Prefixes
        for p in sorted(prefixes):
            token = f"{self.PREFIX_PRE}{p}"
            self.token2id[token] = idx
            idx += 1
        
        # Suffixes
        for s in sorted(suffixes):
            token = f"{self.PREFIX_SUF}{s}"
            self.token2id[token] = idx
            idx += 1
        
        # Endings
        for e in sorted(endings):
            token = f"{self.PREFIX_END}{e}"
            self.token2id[token] = idx
            idx += 1
        
        # Common punctuation and numbers (including Unicode dashes, quotes)
        _PUNCT_CHARS = (
            '.,!?;:-\u2013\u2014\u2015\u2026"\'()[]{}«»/\\@#$%&*+=<>~^|_`№°•·→←↑↓©®™'
            '„\u201c\u201d‟‹›'  # typographic quotes
            '‑‐‒⁃−'  # Unicode dashes/hyphens
            '¡¿'
        )
        for p in _PUNCT_CHARS:
            token = f"{self.PREFIX_PUNCT}{p}"
            if token not in self.token2id:
                self.token2id[token] = idx
                idx += 1
        
        for d in range(10):
            token = f"{self.PREFIX_NUM}{d}"
            self.token2id[token] = idx
            idx += 1
        
        # Character-level fallback tokens (replaces BPE)
        if self.char_fallback or not self.bpe_tokenizer:
            # Cyrillic lowercase + uppercase
            for c in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя':
                for ch in (c, c.upper()):
                    token = f"{self.PREFIX_BPE}{ch}"
                    if token not in self.token2id:
                        self.token2id[token] = idx
                        idx += 1
            # Latin lowercase + uppercase
            for c in 'abcdefghijklmnopqrstuvwxyz':
                for ch in (c, c.upper()):
                    token = f"{self.PREFIX_BPE}{ch}"
                    if token not in self.token2id:
                        self.token2id[token] = idx
                        idx += 1
            # Common symbols
            for ch in ' \t\n_~`^|€£¥°±²³µ¶·¹º¼½¾':
                token = f"{self.PREFIX_BPE}{ch}"
                if token not in self.token2id:
                    self.token2id[token] = idx
                    idx += 1
        elif self.bpe_tokenizer:
            # Legacy BPE mode
            for bpe_id in range(self.bpe_tokenizer.vocab_size):
                piece = self.bpe_tokenizer.sp.id_to_piece(bpe_id)
                token = f"{self.PREFIX_BPE}{piece}"
                if token not in self.token2id:
                    self.token2id[token] = idx
                    idx += 1
        
        # Reverse mapping
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.vocab_size = len(self.token2id)
    
    @classmethod
    def from_tikhonov(
        cls,
        dict_path: str,
        bpe_dir: str | None = None,
        char_fallback: bool = False,
    ) -> "MorphemeTokenizer":
        """Load from Tikhonov JSON file."""
        with open(dict_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        dictionary = data.get("dictionary", data)
        roots_index = data.get("roots_index", {})
        
        bpe_tokenizer = None
        if bpe_dir and not char_fallback:
            from engine.training.kg_bpe_tokenizer import KGBPETokenizer
            bpe_tokenizer = KGBPETokenizer.load(bpe_dir)
        
        # Inject high-frequency function words that Tikhonov omits
        # (he covers derivational morphology, not function/stop words)
        _FUNCTION_WORDS = {
            # === Tikhonov overrides (his entries have multi-variant garbage) ===
            # Pronouns (Tikhonov has "я, мен/я, мн/е" which can't be greedy-parsed)
            'что': 'что', 'это': 'это', 'этот': 'этот',
            'этой': 'этой', 'этого': 'этого', 'этом': 'этом', 'этому': 'этому',
            'этих': 'этих', 'эти': 'эти', 'этим': 'этим', 'эту': 'эту',
            'он': 'он', 'его': 'его', 'ему': 'ему', 'него': 'него',
            'нему': 'нему', 'ним': 'ним', 'нём': 'нём',
            'она': 'она', 'её': 'её', 'ей': 'ей', 'неё': 'неё', 'ней': 'ней',
            'они': 'они', 'них': 'них', 'ним': 'ним', 'ими': 'ими',
            'мне': 'мне', 'меня': 'меня', 'мной': 'мной', 'мною': 'мною',
            'нас': 'нас', 'нам': 'нам', 'нами': 'нами',
            'вам': 'вам', 'вас': 'вас', 'вами': 'вами',
            'тебе': 'тебе', 'тебя': 'тебя', 'тобой': 'тобой',
            'себе': 'себе', 'себя': 'себя', 'собой': 'собой',
            'кто': 'кто', 'кого': 'кого', 'кому': 'кому', 'кем': 'кем',
            'чего': 'чего', 'чему': 'чему', 'чем': 'чем',
            'какой': 'какой', 'каком': 'каком', 'какого': 'какого',
            'какие': 'какие', 'каких': 'каких', 'каким': 'каким',
            'такой': 'такой', 'таком': 'таком', 'такого': 'такого',
            'такие': 'такие', 'таких': 'таких', 'таким': 'таким',
            'весь': 'весь', 'всё': 'всё', 'вся': 'вся',
            'все': 'все', 'всех': 'всех', 'всем': 'всем', 'всего': 'всего',
            'свой': 'свой', 'своя': 'своя', 'своё': 'своё',
            'свои': 'свои', 'своих': 'своих', 'своим': 'своим',
            'своей': 'своей', 'своего': 'своего', 'своему': 'своему',
            'мой': 'мой', 'моя': 'моя', 'моё': 'моё',
            'мои': 'мои', 'моих': 'моих', 'моей': 'моей',
            'наш': 'наш', 'наша': 'наша', 'наше': 'наше',
            'наши': 'наши', 'наших': 'наших', 'нашей': 'нашей',
            'ваш': 'ваш', 'ваша': 'ваша', 'ваше': 'ваше',
            'ваши': 'ваши', 'ваших': 'ваших',
            # Common verbs (Tikhonov: "бы/ть, буд/у" = broken greedy)
            'быть': 'быть', 'был': 'был', 'была': 'была', 'было': 'было',
            'были': 'были', 'будет': 'будет', 'будут': 'будут',
            'будем': 'будем', 'буду': 'буду', 'будете': 'будете',
            'есть': 'есть', 'нет': 'нет',
            'может': 'может', 'могут': 'могут', 'можем': 'можем',
            'хочу': 'хочу', 'хочет': 'хочет', 'хотим': 'хотим',
            'стать': 'стать', 'стал': 'стал', 'стала': 'стала',
            'стали': 'стали', 'станет': 'станет',
            'идти': 'идти', 'идёт': 'идёт', 'идут': 'идут',
            'знать': 'знать', 'знает': 'знает', 'знаю': 'знаю',
            # === 1-letter prepositions, conjunctions, particles ===
            'в': 'в', 'и': 'и', 'а': 'а', 'у': 'у', 'с': 'с',
            'к': 'к', 'о': 'о', 'я': 'я', 'б': 'б',
            # Conjunctions, particles, prepositions, adverbs
            'не': 'не', 'ни': 'ни', 'но': 'но', 'да': 'да', 'до': 'до',
            'же': 'же', 'ли': 'ли', 'бы': 'бы', 'ну': 'ну', 'ка': 'ка',
            'как': 'как', 'так': 'так', 'или': 'или', 'можно': 'можно',
            'если': 'если', 'только': 'только', 'очень': 'очень',
            'когда': 'когда', 'также': 'также', 'без': 'без',
            'сегодня': 'сегодня', 'почему': 'почему', 'сейчас': 'сейчас',
            'всегда': 'всегда', 'именно': 'именно', 'даже': 'даже',
            'где': 'где', 'вот': 'вот', 'ещё': 'ещё', 'еще': 'еще',
            'тоже': 'тоже', 'уже': 'уже', 'либо': 'либо', 'между': 'между',
            'через': 'через', 'после': 'после', 'около': 'около',
            'кроме': 'кроме', 'однако': 'однако', 'итак': 'итак',
            'тогда': 'тогда', 'затем': 'затем', 'зачем': 'зачем',
            'хотя': 'хотя', 'чтобы': 'чтобы', 'пока': 'пока',
            'более': 'более', 'менее': 'менее', 'вообще': 'вообще',
            'нельзя': 'нельзя', 'надо': 'надо', 'нужно': 'нужно',
            'значит': 'значит', 'конечно': 'конечно', 'совсем': 'совсем',
            'просто': 'просто', 'точно': 'точно',
            'лишь': 'лишь', 'там': 'там', 'тут': 'тут',
            'здесь': 'здесь', 'ведь': 'ведь', 'при': 'при',
            'над': 'над', 'про': 'про', 'под': 'под',
            'потому': 'потому', 'поэтому': 'поэтому',
            'вместо': 'вместо', 'вместе': 'вместе',
            # "лет" — broken in Tikhonov (лёт vs год/лет)
            'лет': 'лет',
            # === Borrowings / tech terms ===
            'интернет': 'интернет', 'веб': 'веб', 'сайт': 'сайт',
            'блог': 'блог', 'клик': 'клик', 'чат': 'чат',
            'онлайн': 'онлайн', 'офлайн': 'офлайн',
            'контент': 'контент', 'бренд': 'бренд', 'тренд': 'тренд',
            'бизнес': 'бизнес', 'менеджер': 'менеджер',
            'маркетинг': 'маркетинг', 'дизайн': 'дизайн',
            'сервис': 'сервис', 'сервер': 'сервер',
            'софт': 'софт', 'хард': 'хард', 'код': 'код',
            'фреймворк': 'фреймворк', 'стартап': 'стартап',
            'видео': 'видео', 'аудио': 'аудио', 'фото': 'фото',
            # === Tikhonov multi-variant overrides (greedy-breaking) ===
            'того': 'того', 'тот': 'тот', 'тому': 'тому',
            'той': 'той', 'тем': 'тем', 'те': 'те', 'тех': 'тех',
            'иногда': 'иногда',
            'сми': 'сми', 'млн': 'млн',
            # === Proper nouns ===
            'россия': 'росси', 'хабр': 'хабр', 'яндекс': 'яндекс',
            'google': 'google', 'apple': 'apple', 'linux': 'linux',
        }
        for word, root in _FUNCTION_WORDS.items():
            # Always override — Tikhonov's multi-variant entries break greedy
            ending = word[len(root):] if word != root else ''
            dictionary[word] = {
                'morphemes': [root] + ([ending] if ending else []),
                'root': root,
                'prefixes': [],
                'suffixes': [],
                'ending': ending,
                'raw': f'{root}/{ending}' if ending else f'{root}/',
            }
        
        # === Morpheme-level overrides ===
        # Tikhonov decomposes etymologically (бинарный = bi+nar+н+ый),
        # which is correct historically but unhelpful for NLU.
        # Override with synchronic (modern Russian) decomposition.
        # Format: word → {root, prefixes, suffixes, ending}
        _MORPH_OVERRIDES = {
            # Borrowings with Latin roots — treat as single root in Russian
            'бинарный': {'root': 'бинарн', 'suffixes': [], 'ending': 'ый'},
            'бинарном': {'root': 'бинарн', 'suffixes': [], 'ending': 'ом'},
            'бинарного': {'root': 'бинарн', 'suffixes': [], 'ending': 'ого'},
            'бинарная': {'root': 'бинарн', 'suffixes': [], 'ending': 'ая'},
            'бинарных': {'root': 'бинарн', 'suffixes': [], 'ending': 'ых'},
            'бинарные': {'root': 'бинарн', 'suffixes': [], 'ending': 'ые'},
            # Suppletive verb forms (идти/шёл — different roots)
            # Tikhonov stores infinitive дойти=до+й+ти, root='й' — useless.
            # Override ALL prefixed шёл/шла/шли forms.
            'шёл': {'root': 'шёл', 'suffixes': [], 'ending': ''},
            'шла': {'root': 'шл', 'suffixes': [], 'ending': 'а'},
            'шло': {'root': 'шл', 'suffixes': [], 'ending': 'о'},
            'шли': {'root': 'шл', 'suffixes': [], 'ending': 'и'},
            'дошёл': {'root': 'шёл', 'prefixes': ['до'], 'suffixes': [], 'ending': ''},
            'дошла': {'root': 'шл', 'prefixes': ['до'], 'suffixes': [], 'ending': 'а'},
            'дошли': {'root': 'шл', 'prefixes': ['до'], 'suffixes': [], 'ending': 'и'},
            'пришёл': {'root': 'шёл', 'prefixes': ['при'], 'suffixes': [], 'ending': ''},
            'пришла': {'root': 'шл', 'prefixes': ['при'], 'suffixes': [], 'ending': 'а'},
            'пришло': {'root': 'шл', 'prefixes': ['при'], 'suffixes': [], 'ending': 'о'},
            'пришли': {'root': 'шл', 'prefixes': ['при'], 'suffixes': [], 'ending': 'и'},
            'ушёл': {'root': 'шёл', 'prefixes': ['у'], 'suffixes': [], 'ending': ''},
            'ушла': {'root': 'шл', 'prefixes': ['у'], 'suffixes': [], 'ending': 'а'},
            'ушли': {'root': 'шл', 'prefixes': ['у'], 'suffixes': [], 'ending': 'и'},
            'вышел': {'root': 'шел', 'prefixes': ['вы'], 'suffixes': [], 'ending': ''},
            'вышла': {'root': 'шл', 'prefixes': ['вы'], 'suffixes': [], 'ending': 'а'},
            'вышли': {'root': 'шл', 'prefixes': ['вы'], 'suffixes': [], 'ending': 'и'},
            'пошёл': {'root': 'шёл', 'prefixes': ['по'], 'suffixes': [], 'ending': ''},
            'пошла': {'root': 'шл', 'prefixes': ['по'], 'suffixes': [], 'ending': 'а'},
            'пошли': {'root': 'шл', 'prefixes': ['по'], 'suffixes': [], 'ending': 'и'},
            'зашёл': {'root': 'шёл', 'prefixes': ['за'], 'suffixes': [], 'ending': ''},
            'зашла': {'root': 'шл', 'prefixes': ['за'], 'suffixes': [], 'ending': 'а'},
            'зашли': {'root': 'шл', 'prefixes': ['за'], 'suffixes': [], 'ending': 'и'},
            'нашёл': {'root': 'шёл', 'prefixes': ['на'], 'suffixes': [], 'ending': ''},
            'нашла': {'root': 'шл', 'prefixes': ['на'], 'suffixes': [], 'ending': 'а'},
            'нашли': {'root': 'шл', 'prefixes': ['на'], 'suffixes': [], 'ending': 'и'},
            'подошёл': {'root': 'шёл', 'prefixes': ['подо'], 'suffixes': [], 'ending': ''},
            'подошла': {'root': 'шл', 'prefixes': ['подо'], 'suffixes': [], 'ending': 'а'},
            'подошли': {'root': 'шл', 'prefixes': ['подо'], 'suffixes': [], 'ending': 'и'},
            'обошёл': {'root': 'шёл', 'prefixes': ['обо'], 'suffixes': [], 'ending': ''},
            'обошла': {'root': 'шл', 'prefixes': ['обо'], 'suffixes': [], 'ending': 'а'},
            'обошли': {'root': 'шл', 'prefixes': ['обо'], 'suffixes': [], 'ending': 'и'},
            'перешёл': {'root': 'шёл', 'prefixes': ['пере'], 'suffixes': [], 'ending': ''},
            'перешла': {'root': 'шл', 'prefixes': ['пере'], 'suffixes': [], 'ending': 'а'},
            'перешли': {'root': 'шл', 'prefixes': ['пере'], 'suffixes': [], 'ending': 'и'},
            'прошёл': {'root': 'шёл', 'prefixes': ['про'], 'suffixes': [], 'ending': ''},
            'прошла': {'root': 'шл', 'prefixes': ['про'], 'suffixes': [], 'ending': 'а'},
            'прошли': {'root': 'шл', 'prefixes': ['про'], 'suffixes': [], 'ending': 'и'},
            'сошёл': {'root': 'шёл', 'prefixes': ['со'], 'suffixes': [], 'ending': ''},
            'сошла': {'root': 'шл', 'prefixes': ['со'], 'suffixes': [], 'ending': 'а'},
            'сошли': {'root': 'шл', 'prefixes': ['со'], 'suffixes': [], 'ending': 'и'},
            'отошёл': {'root': 'шёл', 'prefixes': ['ото'], 'suffixes': [], 'ending': ''},
            'отошла': {'root': 'шл', 'prefixes': ['ото'], 'suffixes': [], 'ending': 'а'},
            'отошли': {'root': 'шл', 'prefixes': ['ото'], 'suffixes': [], 'ending': 'и'},
            # Short root verbs (бр- → too short without context)
            'разобрал': {'root': 'бра', 'prefixes': ['разо'], 'suffixes': [], 'ending': 'л'},
            'разобрать': {'root': 'бра', 'prefixes': ['разо'], 'suffixes': [], 'ending': 'ть'},
            'собрал': {'root': 'бра', 'prefixes': ['со'], 'suffixes': [], 'ending': 'л'},
            'собрать': {'root': 'бра', 'prefixes': ['со'], 'suffixes': [], 'ending': 'ть'},
            'выбрал': {'root': 'бра', 'prefixes': ['вы'], 'suffixes': [], 'ending': 'л'},
            'выбрать': {'root': 'бра', 'prefixes': ['вы'], 'suffixes': [], 'ending': 'ть'},
            # Tech borrowings
            'архитектурно': {'root': 'архитектур', 'suffixes': ['н'], 'ending': 'о'},
            'архитектурный': {'root': 'архитектур', 'suffixes': ['н'], 'ending': 'ый'},
            'архитектура': {'root': 'архитектур', 'suffixes': [], 'ending': 'а'},
            'инструкция': {'root': 'инструкци', 'suffixes': [], 'ending': 'я'},
            'инструкций': {'root': 'инструкци', 'suffixes': [], 'ending': 'й'},
            'инструкции': {'root': 'инструкци', 'suffixes': [], 'ending': 'и'},
        }
        for word, parts in _MORPH_OVERRIDES.items():
            root = parts['root']
            prefixes = parts.get('prefixes', [])
            suffixes = parts.get('suffixes', [])
            ending = parts.get('ending', '')
            morphemes = prefixes + [root] + suffixes + ([ending] if ending else [])
            dictionary[word] = {
                'morphemes': morphemes,
                'root': root,
                'prefixes': prefixes,
                'suffixes': suffixes,
                'ending': ending,
                'raw': '/'.join(morphemes),
            }
        
        # Load auto-generated extensions (pymorphy3-derived, corpus-specific)
        ext_path = Path(dict_path).parent / 'morpheme_extensions.json'
        if ext_path.exists():
            with open(ext_path, 'r', encoding='utf-8') as f:
                ext_data = json.load(f)
            ext_entries = ext_data.get('entries', {})
            added = 0
            for word, entry in ext_entries.items():
                if word not in dictionary:  # don't override Tikhonov
                    dictionary[word] = entry
                    added += 1
        
        return cls(dictionary, roots_index, bpe_tokenizer, char_fallback=char_fallback)
    
    def _lookup_word(self, word: str) -> dict | None:
        """Look up word in dictionary with fallback to lemma and related forms.
        
        Strategy:
        1. Exact match
        2. Lowercase match
        3. pymorphy3 lemma (all parses)
        4. Strip reflexive -ся/-сь and retry
        5. Try aspect pair (обучать → обучить)
        6. Try derivational forms (verb → noun, adj → noun)
        7. Try prefix-stripping to find root (обучать → учить)
        """
        word_lower = word.lower().strip()
        
        # 1. Direct lookup
        if word_lower in self.dictionary:
            return self.dictionary[word_lower]
        
        if not self._morph:
            return None
        
        # 2. Try all pymorphy3 parses (not just first)
        parses = self._morph.parse(word_lower)
        for p in parses[:3]:  # Top 3 analyses
            lemma = p.normal_form
            if lemma in self.dictionary:
                return self.dictionary[lemma]
        
        # 3. Strip reflexive -ся/-сь and retry
        candidates = []
        base_word = word_lower
        if word_lower.endswith('ся') or word_lower.endswith('сь'):
            base_word = word_lower[:-2]
            candidates.append(base_word)
            for p in self._morph.parse(base_word)[:2]:
                candidates.append(p.normal_form)
        
        # 4. Try aspect pairs (обучать→обучить, записывать→записать)
        for p in parses[:2]:
            lemma = p.normal_form
            base = lemma.rstrip('ся').rstrip('сь')
            # Imperfective → perfective: -ать → -ить, -ывать → -ать
            if base.endswith('ать'):
                candidates.append(base[:-3] + 'ить')
                candidates.append(base[:-3] + 'еть')
            if base.endswith('ывать'):
                candidates.append(base[:-5] + 'ать')
            if base.endswith('ивать'):
                candidates.append(base[:-5] + 'ить')
                candidates.append(base[:-5] + 'ать')
        
        # 5. Try common derivational forms
        for p in parses[:2]:
            lemma = p.normal_form
            if p.tag.POS in ('VERB', 'INFN'):
                stem = lemma.rstrip('ться').rstrip('ть').rstrip('ти').rstrip('чь')
                for suffix in ['ение', 'ание', 'ние', 'тие', 'ить', 'ка', 'ок', '']:
                    candidates.append(stem + suffix)
            elif p.tag.POS in ('ADJF', 'ADJS'):
                stem = lemma.rstrip('ый').rstrip('ий').rstrip('ой')
                for suffix in ['ость', 'о', 'ство', '']:
                    candidates.append(stem + suffix)
        
        # 6. Try prefix-stripping (обучать → об+учать → учить → root уч)
        PREFIXES = ['пере', 'при', 'про', 'под', 'над', 'раз', 'рас',
                     'от', 'об', 'вы', 'до', 'за', 'на', 'по', 'из',
                     'не', 'у', 'с', 'в', 'о']
        VOWELS = set('аеёиоуыэюя')
        for p in parses[:2]:
            lemma = p.normal_form.rstrip('ся').rstrip('сь')
            for prefix in PREFIXES:
                if lemma.startswith(prefix) and len(lemma) > len(prefix) + 2:
                    stripped = lemma[len(prefix):]
                    candidates.append(stripped)
                    # Strip verb ending and try canonical forms
                    stem = stripped.rstrip('ть').rstrip('ти')
                    # Strip trailing vowel to get consonant stem
                    if stem and stem[-1] in VOWELS:
                        stem_no_v = stem[:-1]
                    else:
                        stem_no_v = stem
                    for ending in ['ить', 'ать', 'еть', 'ение', 'ание', 'ка']:
                        candidates.append(stem_no_v + ending)
                        candidates.append(stripped.rstrip('ать').rstrip('ить') + ending)
        
        for c in candidates:
            if c in self.dictionary:
                return self.dictionary[c]
        
        # 7. Fuzzy match: edit distance 1 (typo tolerance)
        # Only for Cyrillic words >= 5 chars to avoid false positives.
        # Generate all strings within edit distance 1 and check dict.
        # ~600 candidates per word × O(1) dict lookup = negligible cost.
        if len(word_lower) >= 5 and re.match(r'^[а-яёА-ЯЁ]+$', word_lower):
            _CYRILLIC = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
            fuzzy_candidates = set()
            w = word_lower
            n = len(w)
            # Deletions: remove one char
            for i in range(n):
                fuzzy_candidates.add(w[:i] + w[i+1:])
            # Substitutions: replace one char
            for i in range(n):
                for c in _CYRILLIC:
                    if c != w[i]:
                        fuzzy_candidates.add(w[:i] + c + w[i+1:])
            # Transpositions: swap adjacent chars
            for i in range(n - 1):
                if w[i] != w[i+1]:
                    fuzzy_candidates.add(w[:i] + w[i+1] + w[i] + w[i+2:])
            # Insertions: add one char (handles missing letter)
            for i in range(n + 1):
                for c in _CYRILLIC:
                    fuzzy_candidates.add(w[:i] + c + w[i:])
            
            # Check all candidates against dictionary (O(1) per lookup)
            for fc in fuzzy_candidates:
                if fc in self.dictionary:
                    return self.dictionary[fc]
            # pymorphy3 lemma fallback — ONLY for deletions + transpositions
            # (few candidates, ~30-50 instead of 2000+, to keep cost low)
            if self._morph:
                # Regenerate just deletions + transpositions (small set)
                small_candidates = set()
                for i in range(n):
                    small_candidates.add(w[:i] + w[i+1:])  # deletions
                for i in range(n - 1):
                    if w[i] != w[i+1]:
                        small_candidates.add(w[:i] + w[i+1] + w[i] + w[i+2:])
                for fc in small_candidates:
                    for p in self._morph.parse(fc)[:1]:
                        lemma = p.normal_form
                        if lemma in self.dictionary:
                            return self.dictionary[lemma]
        
        return None
    
    def tokenize_word(self, word: str) -> list[str]:
        """Tokenize a single word into morpheme tokens."""
        if not word:
            return []
        
        # Check if number
        if re.match(r'^\d+$', word):
            return [f"{self.PREFIX_NUM}{d}" for d in word]
        
        # Mixed alphanumeric (e.g. "300M", "v3", "K8s") → split at boundaries
        if re.search(r'\d', word) and re.search(r'[a-zA-Zа-яёА-ЯЁ]', word):
            # Split into runs of digits vs letters
            parts = re.findall(r'\d+|[a-zA-Zа-яёА-ЯЁ]+', word)
            result = []
            for p in parts:
                if p.isdigit():
                    result.extend(f"{self.PREFIX_NUM}{d}" for d in p)
                else:
                    result.extend(self.tokenize_word(p))
            return result
        
        # Check if punctuation
        if re.match(r'^[^\w\s]+$', word):
            return [f"{self.PREFIX_PUNCT}{c}" for c in word]
        
        # Look up in dictionary
        entry = self._lookup_word(word)
        if entry:
            # Collect all candidate morphemes with types
            parts = []
            for p in entry.get("prefixes", []):
                parts.append((self.PREFIX_PRE, self._clean_morpheme(p)))
            if entry.get("root"):
                parts.append((self.PREFIX_ROOT, self._clean_morpheme(entry['root'])))
            for s in entry.get("suffixes", []):
                cleaned = self._clean_morpheme(s)
                if cleaned:
                    parts.append((self.PREFIX_SUF, cleaned))
            if entry.get("ending"):
                parts.append((self.PREFIX_END, self._clean_morpheme(entry['ending'])))
            
            # Greedy reconstruction: select morphemes that match the word
            word_lower = word.lower()
            selected = []
            pos = 0
            for prefix, morph in parts:
                morph_lower = morph.lower()
                if pos < len(word_lower) and word_lower[pos:pos+len(morph_lower)] == morph_lower:
                    selected.append(f"{prefix}{morph}")
                    pos += len(morph_lower)
                elif pos < len(word_lower) and len(morph_lower) >= 2:
                    # Flex match: try root without last char (ь-drop, vowel alternation)
                    # e.g. root "осень" → "осен" matches "осени"
                    truncated = morph_lower[:-1]
                    if len(truncated) >= 2 and word_lower[pos:pos+len(truncated)] == truncated:
                        selected.append(f"{prefix}{truncated}")
                        pos += len(truncated)
            
            # If greedy covers the full word, use it
            if pos == len(word_lower) and selected:
                return selected
            
            # Partial match: root/prefixes matched but ending differs
            # (inflected form vs lemma, e.g. "мобилей" vs "мобиль")
            if selected and pos > 0:
                tail = word_lower[pos:]  # remaining unmatched part
                
                # Strategy 1: try to match unmatched suffix from parts
                # e.g. entry has suffix "мобиль" but word has "мобилей"
                # → find longest common prefix of suffix with tail
                for prefix_type, morph in parts:
                    if f"{prefix_type}{morph}" in selected:
                        continue  # already matched
                    morph_lower = morph.lower()
                    # Find longest prefix of this morpheme that matches tail
                    match_len = 0
                    for k in range(min(len(morph_lower), len(tail)), 0, -1):
                        if tail[:k] == morph_lower[:k]:
                            match_len = k
                            break
                    if match_len >= 2:  # at least 2 chars = meaningful
                        stem_part = tail[:match_len]
                        inflection = tail[match_len:]
                        selected.append(f"{prefix_type}{stem_part}")
                        pos += match_len
                        tail = inflection
                        break
                
                # Strategy 2: match remaining tail against known endings
                if tail:
                    ending_token = f"{self.PREFIX_END}{tail}"
                    if ending_token in self.token2id:
                        selected.append(ending_token)
                    else:
                        # Try common Russian endings (longest match first)
                        _ENDINGS = [
                            'ейся', 'ями', 'ого', 'ему', 'ами',
                            'ей', 'ов', 'ом', 'ым', 'ых', 'их',
                            'ая', 'ое', 'ую', 'ие', 'ые', 'ой',
                            'ий', 'ём', 'ях', 'юю', 'ею', 'ию',
                            'а', 'о', 'е', 'и', 'у', 'ы', 'я',
                            'й', 'ь', 'ю',
                        ]
                        matched_ending = False
                        for end in _ENDINGS:
                            if tail == end or tail.endswith(end):
                                stem_rem = tail[:-len(end)] if tail != end else ''
                                if stem_rem:
                                    # Remaining stem fragment before ending
                                    stem_token = f"{self.PREFIX_SUF}{stem_rem}"
                                    if stem_token in self.token2id:
                                        selected.append(stem_token)
                                    else:
                                        selected.append(f"{self.PREFIX_SUF}{stem_rem}")
                                selected.append(f"{self.PREFIX_END}{end}")
                                matched_ending = True
                                break
                        
                        if not matched_ending:
                            # Last resort: treat entire tail as ending
                            selected.append(f"{self.PREFIX_END}{tail}")
                
                return selected
            # Otherwise fall through to word-level or char-fallback
        
        # Latin / mixed-script word: treat as single ROOT token (not char-level)
        # This handles tech terms, code identifiers, proper nouns etc.
        if re.match(r'^[a-zA-Z]', word) and len(word) >= 2:
            token = f"{self.PREFIX_ROOT}{word.lower()}"
            if token not in self.token2id:
                # Dynamically add to vocab so it gets a stable ID
                new_id = len(self.token2id)
                self.token2id[token] = new_id
                self.id2token[new_id] = token
                self.vocab_size = len(self.token2id)
            return [token]
        
        # Cyrillic abbreviation: all uppercase, 2-8 chars (ЦМУ, ГРЧЦ, ССОП, РКН)
        # Treat as single ROOT token instead of per-char fallback
        if re.match(r'^[А-ЯЁ]{2,8}$', word):
            token = f"{self.PREFIX_ROOT}{word.lower()}"
            if token not in self.token2id:
                new_id = len(self.token2id)
                self.token2id[token] = new_id
                self.id2token[new_id] = token
                self.vocab_size = len(self.token2id)
            return [token]
        
        # Character-level fallback (default) or BPE fallback (legacy)
        if self.char_fallback or not self.bpe_tokenizer:
            # Per-character tokenization
            return [f"{self.PREFIX_BPE}{c}" for c in word]
        
        # Legacy BPE fallback
        if self.bpe_tokenizer:
            ids = self.bpe_tokenizer.encode(word)
            tokens = []
            for id_ in ids:
                if id_ in (self.bpe_tokenizer.sp.bos_id(),
                          self.bpe_tokenizer.sp.eos_id(),
                          self.bpe_tokenizer.sp.pad_id()):
                    continue
                piece = self.bpe_tokenizer.sp.id_to_piece(id_)
                tokens.append(f"{self.PREFIX_BPE}{piece}")
            return tokens if tokens else [f"{self.PREFIX_BPE}{word}"]
        
        return [f"{self.PREFIX_BPE}{c}" for c in word]
    
    # Morpheme type IDs for type embedding
    TYPE_PAD = 0
    TYPE_ROOT = 1
    TYPE_SUF = 2
    TYPE_END = 3
    TYPE_PRE = 4
    TYPE_BPE = 5
    TYPE_PUNCT = 6
    TYPE_NUM = 7
    TYPE_SPECIAL = 8  # BOS, EOS, SEP, UNK
    NUM_TYPES = 9
    
    def tokenize(self, text: str, word_sep: bool = False) -> list[str]:
        """Tokenize text into morpheme tokens.
        
        Inserts a space character token between words for boundary preservation.
        Args:
            word_sep: if True, insert '<sep>' between words (legacy, not recommended)
        """
        # Normalize Unicode dashes/hyphens to standard ASCII or en-dash
        # so they get proper punct tokens instead of UNK
        text = re.sub(r'[\u2010\u2011\u2012\u2212]', '-', text)  # ‐ ‑ ‒ − → -
        # Normalize typographic quotes to « » which are in vocab
        text = text.replace('\u201e', '\u00ab')  # „ → «
        text = text.replace('\u201c', '\u00ab')  # " → «
        text = text.replace('\u201d', '\u00bb')  # " → »
        text = text.replace('\u201f', '\u00bb')  # ‟ → »
        
        # Split preserving hyphenated compounds: "K8s-платформы" stays together
        parts = re.findall(r'[\w]+(?:-[\w]+)*|[^\w\s]+|\s+', text)
        
        tokens = []
        for part in parts:
            if part.isspace():
                # Insert one space token per whitespace boundary
                tokens.append(f"{self.PREFIX_BPE} ")
                continue
            # Handle hyphenated compounds: split on hyphens, tokenize each
            if '-' in part and re.match(r'[\w]', part):
                subparts = part.split('-')
                for j, sp in enumerate(subparts):
                    if j > 0:
                        tokens.append(f"{self.PREFIX_PUNCT}-")
                    if sp:
                        tokens.extend(self.tokenize_word(sp))
                continue
            tokens.extend(self.tokenize_word(part))
        
        return tokens
    
    def encode(self, text: str, word_sep: bool = False) -> list[int]:
        """Encode text to token IDs.
        
        If a morpheme token (E:, S:, R:, P:) is not in vocab, decompose
        its content to per-character B: tokens instead of producing UNK.
        
        Args:
            word_sep: if True, insert SEP token between words
        """
        tokens = self.tokenize(text, word_sep=word_sep)
        ids = [self.BOS]
        for t in tokens:
            if t == '<sep>':
                ids.append(self.SEP)
            elif t in self.token2id:
                ids.append(self.token2id[t])
            else:
                # Token not in vocab — decompose to char-level
                # Extract the morpheme content after prefix (X:)
                content = t[2:] if len(t) > 2 and t[1] == ':' else t
                for c in content:
                    # Route digits to N: prefix
                    if c.isdigit():
                        char_token = f"{self.PREFIX_NUM}{c}"
                    else:
                        char_token = f"{self.PREFIX_BPE}{c}"
                    ids.append(self.token2id.get(char_token, self.UNK))
        ids.append(self.EOS)
        return ids
    
    def get_type_id(self, token_id: int) -> int:
        """Get morpheme type ID for a token, for type embedding.
        
        Returns one of TYPE_ROOT, TYPE_SUF, TYPE_END, TYPE_PRE, TYPE_BPE,
        TYPE_PUNCT, TYPE_NUM, TYPE_SPECIAL.
        """
        if token_id in (self.PAD, self.BOS, self.EOS, self.UNK, self.SEP):
            return self.TYPE_SPECIAL
        token = self.id2token.get(token_id, '')
        if token.startswith(self.PREFIX_ROOT): return self.TYPE_ROOT
        if token.startswith(self.PREFIX_SUF):  return self.TYPE_SUF
        if token.startswith(self.PREFIX_END):  return self.TYPE_END
        if token.startswith(self.PREFIX_PRE):  return self.TYPE_PRE
        if token.startswith(self.PREFIX_BPE):  return self.TYPE_BPE
        if token.startswith(self.PREFIX_PUNCT): return self.TYPE_PUNCT
        if token.startswith(self.PREFIX_NUM):  return self.TYPE_NUM
        return self.TYPE_SPECIAL
    
    def get_type_ids(self, token_ids: list[int]) -> list[int]:
        """Get morpheme type IDs for a sequence of tokens."""
        return [self.get_type_id(tid) for tid in token_ids]
    
    # Punctuation that should NOT have space before them
    _NO_SPACE_BEFORE = set(',.!?;:)]}»"\'…')
    # Punctuation that should NOT have space after them  
    _NO_SPACE_AFTER = set('([{«"\'')
    
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text.
        
        Structural rules:
        - Space before ROOT/PRE tokens (new word)
        - No space before ,  .  !  ?  ;  :  )  ]
        - No space after  (  [  «  "
        - Explicit B:' ' tokens produce spaces
        """
        parts = []
        prev_was_space = True  # suppress leading space
        prev_no_space_after = False  # e.g. after opening bracket
        prev_prefix = None  # track morpheme type for word continuity
        for id_ in ids:
            if id_ in (self.PAD, self.BOS, self.EOS, self.UNK, self.SEP):
                continue
            token = self.id2token.get(id_, "<unk>")
            
            # Detect type and strip prefix
            cur_prefix = None
            text = token
            for prefix in (self.PREFIX_ROOT, self.PREFIX_SUF, self.PREFIX_END,
                          self.PREFIX_PRE, self.PREFIX_BPE, self.PREFIX_PUNCT,
                          self.PREFIX_NUM):
                if token.startswith(prefix):
                    cur_prefix = prefix
                    text = token[len(prefix):]
                    break
            
            # Punctuation: no space before closing/terminal punct
            is_no_space_before = (
                cur_prefix == self.PREFIX_PUNCT 
                and text and text[0] in self._NO_SPACE_BEFORE
            )
            
            # No auto-space insertion — tokenize() now explicitly adds B:' '
            # for word boundaries. Just handle punctuation spacing.
            if is_no_space_before:
                # Remove trailing space before closing punct
                if parts and parts[-1] == ' ':
                    parts.pop()
            
            parts.append(text)
            prev_was_space = (text == ' ')
            prev_no_space_after = (
                cur_prefix == self.PREFIX_PUNCT
                and text and text[-1] in self._NO_SPACE_AFTER
            )
            prev_prefix = cur_prefix
        return ''.join(parts)
    
    def stats(self) -> dict:
        """Return vocabulary statistics."""
        roots = sum(1 for k in self.token2id if k.startswith(self.PREFIX_ROOT))
        suffixes = sum(1 for k in self.token2id if k.startswith(self.PREFIX_SUF))
        endings = sum(1 for k in self.token2id if k.startswith(self.PREFIX_END))
        prefixes = sum(1 for k in self.token2id if k.startswith(self.PREFIX_PRE))
        bpe = sum(1 for k in self.token2id if k.startswith(self.PREFIX_BPE))
        
        return {
            "vocab_size": self.vocab_size,
            "roots": roots,
            "prefixes": prefixes,
            "suffixes": suffixes,
            "endings": endings,
            "bpe_fallback": bpe,
            "special": 5,
            "dictionary_words": len(self.dictionary),
        }


def main():
    """Quick demo and benchmark."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dict', required=True, help='Tikhonov JSON file')
    parser.add_argument('--bpe-dir', default=None, help='BPE tokenizer directory')
    parser.add_argument('--text', default=None, help='Text to tokenize')
    args = parser.parse_args()
    
    print("Loading MorphemeTokenizer...")
    tok = MorphemeTokenizer.from_tikhonov(args.dict, args.bpe_dir)
    
    stats = tok.stats()
    print(f"\nVocabulary stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}")
    
    # Demo sentences
    sentences = [
        "Программирование на Python является одним из самых популярных направлений",
        "Москва — столица Российской Федерации и крупнейший город страны",
        "Нейронные сети обучаются методом обратного распространения ошибки",
        "Перепрограммировать запрограммированный компьютер непросто",
    ]
    
    if args.text:
        sentences = [args.text]
    
    print(f"\n{'='*70}")
    for sent in sentences:
        tokens = tok.tokenize(sent)
        ids = tok.encode(sent)
        
        # Count morpheme types
        n_root = sum(1 for t in tokens if t.startswith(tok.PREFIX_ROOT))
        n_suf = sum(1 for t in tokens if t.startswith(tok.PREFIX_SUF))
        n_end = sum(1 for t in tokens if t.startswith(tok.PREFIX_END))
        n_pre = sum(1 for t in tokens if t.startswith(tok.PREFIX_PRE))
        n_bpe = sum(1 for t in tokens if t.startswith(tok.PREFIX_BPE))
        n_words = len(sent.split())
        
        print(f"\n  Input: {sent}")
        print(f"  Tokens ({len(tokens)}): {tokens}")
        print(f"  Morpheme mix: {n_root}R + {n_pre}P + {n_suf}S + {n_end}E + {n_bpe}BPE")
        print(f"  Coverage: {(len(tokens)-n_bpe)/max(len(tokens),1)*100:.0f}% morpheme, {n_bpe}/{len(tokens)} BPE fallback")
        print(f"  Fertility: {len(tokens)/n_words:.2f} tokens/word")
        
        # Decode roundtrip
        decoded = tok.decode(ids)
        print(f"  Decoded: {decoded}")


if __name__ == '__main__':
    main()
