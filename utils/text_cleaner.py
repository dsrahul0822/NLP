# utils/text_cleaner.py
from __future__ import annotations
import re
from typing import List
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import nltk

BASIC_PUNCT_RE = re.compile(r"[^a-zA-Z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

def _get_stopwords():
    sw = set(ENGLISH_STOP_WORDS)
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        try:
            nltk.download("stopwords", quiet=True)
        except Exception:
            pass
    try:
        from nltk.corpus import stopwords as nltk_sw
        sw.update(nltk_sw.words("english"))
    except Exception:
        pass
    return sw

class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Sklearn-compatible cleaner (picklable):
    - lowercase
    - remove punctuation & numbers
    - remove stopwords
    - (optional) Porter stemming
    - min token length filter
    """
    def __init__(
        self,
        lowercase: bool = True,
        remove_punct_num: bool = True,
        remove_stopwords: bool = True,
        use_stemming: bool = False,
        min_word_len: int = 2,
    ):
        self.lowercase = lowercase
        self.remove_punct_num = remove_punct_num
        self.remove_stopwords = remove_stopwords
        self.use_stemming = use_stemming
        self.min_word_len = int(min_word_len)
        self._stemmer = None
        self._stopwords = None

    def fit(self, X, y=None):
        if self.remove_stopwords and self._stopwords is None:
            self._stopwords = _get_stopwords()
        if self.use_stemming and self._stemmer is None:
            try:
                from nltk.stem.porter import PorterStemmer
                self._stemmer = PorterStemmer()
            except Exception:
                self._stemmer = None
        return self

    def transform(self, X):
        if isinstance(X, pd.Series):
            arr = X.astype(str).fillna("").tolist()
        elif isinstance(X, np.ndarray):
            arr = ["" if v is None else str(v) for v in X.tolist()]
        else:
            arr = ["" if v is None else str(v) for v in X]

        out: List[str] = []
        for t in arr:
            if self.lowercase:
                t = t.lower()
            if self.remove_punct_num:
                t = BASIC_PUNCT_RE.sub(" ", t)
            t = MULTI_SPACE_RE.sub(" ", t).strip()

            toks = [tok for tok in t.split() if len(tok) >= self.min_word_len]
            if self.remove_stopwords and self._stopwords:
                toks = [tok for tok in toks if tok not in self._stopwords]
            if self.use_stemming and self._stemmer:
                toks = [self._stemmer.stem(tok) for tok in toks]
            out.append(" ".join(toks))
        return out
