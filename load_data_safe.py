
import pandas as pd
import streamlit as st

@st.cache_data
def load_data_safe():
    error_messages = []
    for sep in [',', ';', '\t']:
        for encoding in ['utf-8', 'latin-1']:
            try:
                df = pd.read_csv("spam_NLP.csv", sep=sep, encoding=encoding, quotechar='"', on_bad_lines='skip')
                if df.shape[1] >= 2:
                    df = df.iloc[:, :2]
                    df.columns = ['label', 'text']
                    return df
            except Exception as e:
                error_messages.append(f"Separator: '{sep}', Encoding: '{encoding}'\n{e}")
    st.error("❌ Nie udało się wczytać pliku CSV żadną z metod.")
    st.text("\n\n".join(error_messages))
    st.stop()
