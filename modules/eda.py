import matplotlib
matplotlib.use('Agg')  

import matplotlib.pyplot as plt
import seaborn as sns
import os

PLOT_FOLDER = "static/plots"

# Ensure folder exists
os.makedirs(PLOT_FOLDER, exist_ok=True)


def generate_summary(df):
    return df.describe(include='all').to_html()


def create_plots(df):
    plot_paths = []

    # 1. Correlation heatmap (numeric only)
    numeric_df = df.select_dtypes(include=['int64', 'float64'])

    if not numeric_df.empty:
        plt.figure(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")

        heatmap_path = os.path.join(PLOT_FOLDER, "heatmap.png")
        plt.savefig(heatmap_path, bbox_inches='tight')
        plt.close()

        plot_paths.append("/" + heatmap_path)

    # 2. Histograms (first 3 numeric columns only)
    for col in numeric_df.columns[:3]:
        plt.figure()
        sns.histplot(df[col], kde=True)

        path = os.path.join(PLOT_FOLDER, f"{col}_hist.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

        plot_paths.append("/" + path)

    # 3. Count plot (first categorical column)
    cat_cols = df.select_dtypes(include=['object']).columns

    if len(cat_cols) > 0:
        col = cat_cols[0]

        plt.figure()
        df[col].value_counts().head(10).plot(kind='bar')

        path = os.path.join(PLOT_FOLDER, f"{col}_count.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()

        plot_paths.append("/" + path)

    return plot_paths