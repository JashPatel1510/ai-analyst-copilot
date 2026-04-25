import pandas as pd

def generate_insights(df):
    insights = []
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    # 1. Missing value insights
    missing = df.isnull().sum()
    for col, val in missing.items():
        if val > 0:
            insights.append(f"Column '{col}' has {val} missing values ({round(val/len(df)*100,1)}% of rows).")

    # 2. Correlation (deduplicated, no A→B and B→A both)
    if not numeric_df.empty:
        corr = numeric_df.corr()
        seen = set()
        for col in corr.columns:
            for row in corr.index:
                if col == row:
                    continue
                pair = tuple(sorted([col, row]))
                if pair in seen:
                    continue
                seen.add(pair)
                value = corr.loc[row, col]
                if abs(value) > 0.7:
                    direction = "positively" if value > 0 else "negatively"
                    insights.append(f"'{pair[0]}' and '{pair[1]}' are strongly {direction} correlated ({round(value, 2)}).")

    # 3. Skew detection (top 3 most skewed only)
    if not numeric_df.empty:
        skews = numeric_df.skew().abs().sort_values(ascending=False)
        top_skewed = skews[skews > 1].head(3)
        for col, skew_val in top_skewed.items():
            insights.append(f"'{col}' is highly skewed (skew={round(skew_val,2)}) — consider log transformation.")

    # 4. Outlier-prone columns (top 3 by CV)
    if not numeric_df.empty:
        cv = (numeric_df.std() / numeric_df.mean().replace(0, float('nan'))).abs()
        top_cv = cv.sort_values(ascending=False).head(3)
        for col, val in top_cv.items():
            if val > 1:
                insights.append(f"'{col}' has high variability (CV={round(val,2)}) — possible outliers present.")

    # 5. Categorical dominance
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        top = df[col].value_counts().idxmax()
        perc = df[col].value_counts(normalize=True).max()
        if perc > 0.5:
            insights.append(f"'{col}' is dominated by '{top}' ({round(perc*100,1)}% of rows) — low diversity.")

    # 6. Class imbalance check (low cardinality columns)
    for col in numeric_df.columns:
        if df[col].nunique() <= 5:
            counts = df[col].value_counts(normalize=True)
            if counts.max() > 0.75:
                insights.append(f"'{col}' may be imbalanced — dominant class is {round(counts.max()*100,1)}% of data.")

    # Fallback
    if not insights:
        insights.append("No strong patterns detected. Dataset appears balanced and clean.")

    return insights