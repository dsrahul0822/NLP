# utils/shared.py
from __future__ import annotations
from pathlib import Path
import pandas as pd
import streamlit as st

DEFAULT_PATH = "/mnt/data/Restaurant_Reviews.tsv"

def _read_any_delim(path_or_buffer):
    try:
        return pd.read_csv(path_or_buffer, sep=None, engine="python")
    except Exception:
        try:
            return pd.read_csv(path_or_buffer, sep="\t")
        except Exception:
            return pd.read_csv(path_or_buffer)

def load_default_dataset():
    p = Path(DEFAULT_PATH)
    if p.exists():
        return _read_any_delim(p)
    return None

def set_dataset(df: pd.DataFrame):
    st.session_state._dataset = df

def get_dataset() -> pd.DataFrame | None:
    return st.session_state.get("_dataset")

def set_columns(text_col: str, label_col: str):
    st.session_state._text_col = text_col
    st.session_state._label_col = label_col

def get_columns():
    return st.session_state.get("_text_col"), st.session_state.get("_label_col")

def ensure_dataset_loaded():
    if get_dataset() is None:
        df = load_default_dataset()
        if df is not None:
            set_dataset(df)
