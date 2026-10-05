
import streamlit as st
import pandas as pd
from pathlib import Path


DATA_FILE = "english_learning_dataset_2000.xlsx"


@st.cache_data
def load_data():
    path = Path(DATA_FILE)

    if not path.exists():
        st.error(f"Dataset not found: {DATA_FILE}")
        st.stop()

    df = pd.read_excel(path)

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df

