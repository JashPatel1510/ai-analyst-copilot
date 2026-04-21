import pandas as pd

def generate_insights(df):
    insights=[]
    
    # 1. missing value insights
    missing = df.isnull().sum()
    for col, val in missing.items():
        if val > 0:
            insights.append(f"Column '{col}' has {val} missing values (data quality issue).")
            
            
    # 2. Correlation insight (numeric only)
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if not numeric_df.empty:
        corr = numeric_df.corr()

        for col in corr.columns:
            for row in corr.index:
                value = corr.loc[row, col]

                if col != row and abs(value) > 0.7:
                    insights.append(f"Strong relationship: '{row}' and '{col}' (correlation = {round(value,2)})")


    # 3. Skew detection
    for col in numeric_df.columns:
        skew = numeric_df[col].skew()
        if abs(skew) > 1:
            insights.append(f"Column '{col}' is highly skewed (not normally distributed).")

    # 4. High variance columns
    for col in numeric_df.columns:
        if numeric_df[col].std() > numeric_df[col].mean():
            insights.append(f"Column '{col}' has high variation (possible outliers).")

    # 5. Categorical dominance
    cat_cols = df.select_dtypes(include=['object']).columns

    for col in cat_cols:
        top = df[col].value_counts().idxmax()
        perc = df[col].value_counts(normalize=True).max()

        if perc > 0.5:
            insights.append(f"'{col}' is dominated by '{top}' ({round(perc*100,1)}% of data).")

    # fallback
    if not insights:
        insights.append("No strong patterns detected. Dataset may be balanced or requires deeper analysis.")

    return insights