import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    roc_auc_score
)


def target_distribution(df, target_col):

    counts = df[target_col].value_counts()

    fig, ax = plt.subplots(
        figsize=(8, 4)
    )

    counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_title(
        "Observed Learning Progress Track Distribution"
    )

    ax.set_xlabel(
        "Learning Progress Track"
    )

    ax.set_ylabel(
        "Number of Learners"
    )

    plt.xticks(rotation=0)
    plt.tight_layout()

    return fig


def accuracy_chart(results_df):

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        results_df["Model"],
        results_df["Accuracy"] * 100
    )

    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Model")

    ax.set_title(
        "Machine Learning Model Accuracy Comparison"
    )

    ax.set_ylim(0, 100)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    return fig


def metrics_chart(results_df):

    metrics = [
        "Accuracy",
        "Precision",
        "Recall",
        "Weighted F1",
        "Macro F1"
    ]

    fig, ax = plt.subplots(
        figsize=(12, 5)
    )

    x = range(len(results_df))
    width = 0.15

    for i, metric in enumerate(metrics):

        ax.bar(
            [
                value + (i - 2) * width
                for value in x
            ],
            results_df[metric] * 100,
            width,
            label=metric
        )

    ax.set_xticks(list(x))

    ax.set_xticklabels(
        results_df["Model"],
        rotation=20,
        ha="right"
    )

    ax.set_ylabel("Score (%)")
    ax.set_ylim(0, 100)

    ax.set_title(
        "Comprehensive Model Performance Comparison"
    )

    ax.legend()

    plt.tight_layout()

    return fig


def confusion_matrix_chart(
    y_test,
    predictions,
    classes,
    model_name
):

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=classes
    )

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax
    )

    ax.set_title(
        f"Confusion Matrix — {model_name}"
    )

    ax.set_xlabel(
        "Predicted Track"
    )

    ax.set_ylabel(
        "Actual Track"
    )

    plt.tight_layout()

    return fig


def roc_chart(
    y_test,
    probabilities,
    classes
):

    fig, ax = plt.subplots(
        figsize=(7, 5)
    )

    for i, class_name in enumerate(classes):

        class_true = (
            y_test == class_name
        ).astype(int)

        probability = probabilities[:, i]

        fpr, tpr, _ = roc_curve(
            class_true,
            probability
        )

        auc = roc_auc_score(
            class_true,
            probability
        )

        ax.plot(
            fpr,
            tpr,
            linewidth=2,
            label=f"{class_name} (AUC = {auc:.2f})"
        )

    ax.plot(
        [0, 1],
        [0, 1],
        "k--",
        label="Random Baseline"
    )

    ax.set_xlabel(
        "False Positive Rate"
    )

    ax.set_ylabel(
        "True Positive Rate"
    )

    ax.set_title(
        "Multi-Class ROC-AUC Performance"
    )

    ax.legend(
        loc="lower right",
        fontsize=8
    )

    plt.tight_layout()

    return fig