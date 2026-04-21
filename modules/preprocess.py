import pandas as pd

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):

    # Fix numeric columns
    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].mean())

    # Fix categorical columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].fillna("Unknown")

    return df