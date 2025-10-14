import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from src.paths import ProjectPaths

# === Configuration ===
FIGURES_OUT = ProjectPaths.REPORTS_FIGURES_FOLDER
TABLES_OUT = ProjectPaths.REPORTS_TABLES_FOLDER
FIG_DPI = 150
BOX_KW = dict(showfliers=False)
VIOLIN_KW = dict(density_norm='width', cut=0, inner='quartile')


def log(message):
    print(f"[LOG] {message}")


# === Label Distribution ===
def plot_label_distribution(df):
    df_path = ProjectPaths.DATA_ML_CSV_FOLDER / "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv"
    df_title = df_path.stem.split(".pcap")[0].replace("-", " ").strip()

    label_counts = df['Label'].value_counts()
    palette = sns.color_palette("Set2", n_colors=len(label_counts))
    color_map = {label: palette[i] for i, label in enumerate(label_counts.index)}
    colors = [color_map[label] for label in label_counts.index]

    plt.figure(figsize=(6, 6))
    plt.pie(
        label_counts.values.astype(int),
        labels=label_counts.index,
        colors=colors,
        autopct=lambda pct: f'{int(round(pct))}%',
        startangle=140
    )
    plt.title(f'{df_title}\nTraffic Label Distribution\n({len(df)} entries)')
    plt.axis('equal')
    plt.tight_layout()
    out_path = FIGURES_OUT / "label_distribution_pie.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    log(f"Saved label distribution pie chart: {out_path}")


# === SVD Projection Plot ===
def plot_svd_projection(train_svd_df):
    ddos_mean_y = train_svd_df[train_svd_df['Label'] == 'DDoS']['SVD2'].mean()
    global_mean_x = train_svd_df['SVD1'].mean()
    global_mean_y = train_svd_df['SVD2'].mean()

    flip_y = ddos_mean_y > global_mean_y
    flip_x = global_mean_x < 0

    train_svd_df_plot = train_svd_df.copy()
    if flip_y:
        train_svd_df_plot['SVD2'] *= -1
    if flip_x:
        train_svd_df_plot['SVD1'] *= -1

    plt.figure(figsize=(10, 6))
    for label in train_svd_df_plot['Label'].unique():
        subset = train_svd_df_plot[train_svd_df_plot['Label'] == label]
        plt.scatter(subset['SVD1'], subset['SVD2'], label=label, alpha=0.25, s=8)

    plt.title('SVD Projection: Behavioral Clustering')
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    plt.legend(frameon=False)
    plt.box(False)
    plt.grid(False)
    plt.tight_layout()
    out_path = FIGURES_OUT / "svd_projection_behavioral_clustering.png"
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    log(f"Saved SVD projection figure: {out_path}")


# === Feature Importance and Stats ===
def save_svd_feature_weights(components_df):
    components_df.to_csv(TABLES_OUT / "svd_components.csv")
    abs_weights = components_df.abs()
    top4_svd1 = abs_weights.loc['SVD1'].sort_values(ascending=False).head(4).index.tolist()
    top4_svd2 = abs_weights.loc['SVD2'].sort_values(ascending=False).head(4).index.tolist()

    pd.Series(top4_svd1, name="top4_svd1_features").to_csv(TABLES_OUT / "top4_svd1_features.csv", index=False)
    pd.Series(top4_svd2, name="top4_svd2_features").to_csv(TABLES_OUT / "top4_svd2_features.csv", index=False)

    return top4_svd1, top4_svd2


def compute_feature_summary(plot_df, features, filename):
    rows = []
    for feat in features:
        grp = plot_df.groupby('Label')[feat]
        stats = grp.agg(['count', 'median', 'mean', 'std']).reset_index()
        stats.insert(0, "feature", feat)
        rows.append(stats)
    df_stats = pd.concat(rows, ignore_index=True)
    df_stats.to_csv(TABLES_OUT / filename, index=False)


def compute_feature_details(plot_df, features, filename):
    rows = []
    for feat in features:
        s = plot_df[feat]
        pct = s.quantile([0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0]).to_dict()
        zeros = int((s == 0).sum())
        skew_val = float(s.skew())
        rows.append({
            "feature": feat,
            "zeros": zeros,
            "skew": skew_val,
            "min": pct[0.0],
            "p01": pct[0.01],
            "p05": pct[0.05],
            "p25": pct[0.25],
            "p50": pct[0.5],
            "p75": pct[0.75],
            "p90": pct[0.9],
            "p99": pct[0.99],
            "max": pct[1.0]
        })
    df_detail = pd.DataFrame(rows)
    df_detail.to_csv(TABLES_OUT / filename, index=False)


# === Feature Distribution Plots ===
def plot_feature_group(plot_df, features, title_prefix, filename):
    n = len(features)
    fig, axes = plt.subplots(nrows=n, ncols=2, figsize=(10, 3 * n), dpi=FIG_DPI,
                             gridspec_kw={'width_ratios': [1, 1]})
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, feat in enumerate(features):
        ax_box = axes[i, 0]
        ax_violin = axes[i, 1]

        sns.boxplot(x='Label', y=feat, hue='Label', data=plot_df, ax=ax_box,
                    **BOX_KW, palette="Set2", legend=False)
        ax_box.set_title(f"Box: {feat}")
        ax_box.set_xlabel('')
        ax_box.set_ylabel('')
        if ax_box.get_legend():
            ax_box.get_legend().remove()

        sns.violinplot(x='Label', y=feat, hue='Label', data=plot_df, ax=ax_violin,
                       palette="Set2", legend=False, **VIOLIN_KW)
        ax_violin.set_title(f"Violin: {feat}")
        ax_violin.set_xlabel('')
        ax_violin.set_ylabel('')
        if ax_violin.get_legend():
            ax_violin.get_legend().remove()

        for ax in (ax_box, ax_violin):
            ax.tick_params(axis='both', which='major', labelsize=8)

    fig.suptitle(f"{title_prefix}", fontsize=12)
    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path = FIGURES_OUT / filename
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    log(f"Saved feature distribution plot: {out_path}")


# === Entry Point ===
def run_eda(df, df_train, train_svd_df, components_df):
    plot_label_distribution(df)
    plot_svd_projection(train_svd_df)

    top4_svd1, top4_svd2 = save_svd_feature_weights(components_df)

    numeric_cols = df_train.select_dtypes(include='number').columns
    plot_df = df_train[numeric_cols].copy()
    plot_df['Label'] = df_train['Label'].values

    compute_feature_summary(plot_df, top4_svd1, "top4_svd1_feature_stats_by_label.csv")
    compute_feature_summary(plot_df, top4_svd2, "top4_svd2_feature_stats_by_label.csv")

    compute_feature_details(plot_df, top4_svd1, "top4_svd1_feature_percentiles.csv")
    compute_feature_details(plot_df, top4_svd2, "top4_svd2_feature_percentiles.csv")

    plot_feature_group(plot_df, top4_svd1, "SVD1: Top Weighted Features", "svd1_top4_features_box_violin.png")
    plot_feature_group(plot_df, top4_svd2, "SVD2: Top Weighted Features", "svd2_top4_features_box_violin.png")

    train_svd_df.to_csv(TABLES_OUT / "train_svd_projection.csv", index=False)
    log(f"All EDA outputs saved to {ProjectPaths.REPORTS_FOLDER.resolve()}")
