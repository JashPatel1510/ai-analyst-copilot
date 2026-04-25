import pandas as pd

def load_data(filepath):
    encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            return df
        except (UnicodeDecodeError, Exception):
            continue
    raise ValueError("Could not read file with any known encoding.")


def clean_data(df):

    # Fix numeric columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Fix categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")

    return df