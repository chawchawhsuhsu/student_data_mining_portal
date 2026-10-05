import os
import ast
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Mining Logs",
    page_icon="⛏️",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULES_FILE = os.path.join(BASE_DIR, "fp_growth_rules.csv")


@st.cache_data
def load_rules():
    if not os.path.exists(RULES_FILE):
        return None

    rules = pd.read_csv(RULES_FILE)

    required = [
        "antecedents",
        "consequents",
        "support",
        "confidence",
        "lift"
    ]

    if not all(column in rules.columns for column in required):
        return None

    for column in ["support", "confidence", "lift"]:
        rules[column] = pd.to_numeric(
            rules[column],
            errors="coerce"
        )

    return rules.dropna(
        subset=["support", "confidence", "lift"]
    )


def format_itemset(value):
    try:
        if isinstance(value, str):
            value = ast.literal_eval(value)

        if isinstance(value, (set, frozenset, list, tuple)):
            return ", ".join(map(str, value))
    except Exception:
        pass

    return str(value)


st.title("⛏️ Pattern Mining")
st.caption(
    "Discover frequent combinations of English-learning characteristics "
    "using FP-Growth association rules."
)

# =========================================================
# FP-GROWTH
# =========================================================
st.header("🔗 FP-Growth Association Rules")

rules = load_rules()

if rules is None:
    st.warning(
        "FP-Growth rules file was not found or has an invalid format."
    )
    st.info(
        "Train FP-Growth and save the results as "
        "`fp_growth_rules.csv`."
    )
else:
    st.write(
        "FP-Growth identifies combinations of learner characteristics "
        "that frequently occur together. The rules describe associations "
        "rather than cause-and-effect relationships."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        min_support = st.slider(
            "Minimum Support",
            0.01,
            0.20,
            0.05,
            0.01
        )

    with col2:
        min_confidence = st.slider(
            "Minimum Confidence",
            0.10,
            0.90,
            0.30,
            0.05
        )

    with col3:
        max_lift = max(1.0, float(rules["lift"].max()))

        min_lift = st.slider(
            "Minimum Lift",
            0.0,
            max_lift,
            1.0,
            0.1
        )

    filtered = rules[
        (rules["support"] >= min_support)
        & (rules["confidence"] >= min_confidence)
        & (rules["lift"] >= min_lift)
    ].copy()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Rules", len(rules))
    c2.metric("Matching Rules", len(filtered))

    if not filtered.empty:
        c3.metric(
            "Highest Lift",
            f"{filtered['lift'].max():.2f}"
        )
    else:
        c3.metric("Highest Lift", "—")

    if filtered.empty:
        st.info("No rules match the selected thresholds.")
    else:
        display_rules = filtered.copy()

        display_rules["IF"] = display_rules[
            "antecedents"
        ].apply(format_itemset)

        display_rules["THEN"] = display_rules[
            "consequents"
        ].apply(format_itemset)

        display_rules["Rule"] = (
            display_rules["IF"]
            + " → "
            + display_rules["THEN"]
        )

        st.subheader("Discovered Associations")

        st.dataframe(
            display_rules[
                ["Rule", "support", "confidence", "lift"]
            ].sort_values(
                "lift",
                ascending=False
            ),
            use_container_width=True,
            hide_index=True
        )

        fig = px.scatter(
            filtered,
            x="support",
            y="confidence",
            size="lift",
            color="lift",
            hover_data=[
                "antecedents",
                "consequents"
            ],
            title="Support vs Confidence of Association Rules"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        strongest = filtered.loc[
            filtered["lift"].idxmax()
        ]

        st.success(
            f"**Strongest association:** "
            f"{format_itemset(strongest['antecedents'])} → "
            f"{format_itemset(strongest['consequents'])} "
            f"(Lift = {strongest['lift']:.2f})"
        )

    with st.expander("ℹ️ How to Interpret the Measures"):
        st.markdown("""
**Support** — How frequently the combination appears in the dataset.

**Confidence** — How often the THEN condition occurs when the IF
condition occurs.

**Lift** — How much stronger the association is compared with
random co-occurrence.

A lift greater than 1 generally indicates a positive association.
These measures show association, not causation.

### Why FP-Growth is included

FP-Growth is part of the descriptive data-mining component because
it discovers recurring combinations of learner characteristics.
It does not predict an individual learner's progress.
""")

st.divider()

st.caption(
    "English Learning Data Mining Portal • Descriptive Pattern Mining • FP-Growth"
)
