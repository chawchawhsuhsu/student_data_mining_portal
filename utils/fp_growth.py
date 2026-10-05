import pandas as pd
from mlxtend.frequent_patterns import (
    association_rules,
    fpgrowth
)


def create_transactions(df, target_col):
    """Convert learning data into FP-Growth transactions."""

    data = df.drop(
        columns=[target_col]
    ).copy()

    encoded_columns = []

    for column in data.columns:

        series = data[column]

        if pd.api.types.is_numeric_dtype(series):

            if series.nunique() <= 6:

                encoded = pd.get_dummies(
                    series,
                    prefix=column
                )

            else:

                try:
                    categories = pd.qcut(
                        series,
                        q=3,
                        labels=[
                            "Low",
                            "Medium",
                            "High"
                        ],
                        duplicates="drop"
                    )

                    encoded = pd.get_dummies(
                        categories,
                        prefix=column
                    )

                except Exception:
                    encoded = pd.DataFrame(
                        index=data.index
                    )

        else:

            encoded = pd.get_dummies(
                series.astype(str),
                prefix=column
            )

        encoded_columns.append(encoded)

    if not encoded_columns:
        return pd.DataFrame()

    return pd.concat(
        encoded_columns,
        axis=1
    ).astype(bool)


def mine_rules(
    df,
    target_col,
    min_support,
    min_confidence
):
    """Run FP-Growth and generate association rules."""

    transactions = create_transactions(
        df,
        target_col
    )

    if transactions.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    try:

        itemsets = fpgrowth(
            transactions,
            min_support=min_support,
            use_colnames=True
        )

        if itemsets.empty:
            return (
                itemsets,
                pd.DataFrame()
            )

        rules = association_rules(
            itemsets,
            metric="confidence",
            min_threshold=min_confidence
        )

        return itemsets, rules

    except Exception:
        return (
            pd.DataFrame(),
            pd.DataFrame()
        )


def format_rules(rules):
    """Prepare rules for Streamlit display."""

    if rules.empty:
        return pd.DataFrame()

    result = rules.copy()

    result["Antecedents"] = result[
        "antecedents"
    ].apply(
        lambda x: ", ".join(
            sorted(x)
        )
    )

    result["Consequents"] = result[
        "consequents"
    ].apply(
        lambda x: ", ".join(
            sorted(x)
        )
    )

    result["Support (%)"] = (
        result["support"] * 100
    ).round(2)

    result["Confidence (%)"] = (
        result["confidence"] * 100
    ).round(2)

    result["Lift"] = result[
        "lift"
    ].round(2)

    return result[
        [
            "Antecedents",
            "Consequents",
            "Support (%)",
            "Confidence (%)",
            "Lift"
        ]
    ].sort_values(
        [
            "Confidence (%)",
            "Lift"
        ],
        ascending=False
    )


def rule_statistics(rules):
    """Calculate FP-Growth summary statistics."""

    if rules.empty:
        return 0, 0, 0

    return (
        rules["support"].mean(),
        rules["confidence"].mean(),
        rules["lift"].mean()
    )