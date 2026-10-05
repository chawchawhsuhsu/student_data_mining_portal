MODEL_EXPLANATIONS = {

    "Decision Tree": {
        "role": "Baseline classifier",
        "why_used": (
            "Decision Tree provides an interpretable baseline model "
            "that can represent decision rules between learner "
            "characteristics and learning progress."
        ),
        "strength": (
            "Easy to interpret and explain because predictions "
            "can be represented as a sequence of decision rules."
        ),
        "limitation": (
            "A single decision tree can be sensitive to the training "
            "data and may struggle to represent complex relationships."
        )
    },

    "Random Forest": {
        "role": "Ensemble classifier",
        "why_used": (
            "Random Forest combines multiple decision trees and "
            "aggregates their predictions, providing a more robust "
            "alternative to a single decision tree."
        ),
        "strength": (
            "Generally more stable than a single tree and can capture "
            "different patterns across learner characteristics."
        ),
        "limitation": (
            "Less directly interpretable than a single decision tree "
            "because predictions come from many trees."
        )
    },

    "Gradient Boosting": {
        "role": "Boosting-based classifier",
        "why_used": (
            "Gradient Boosting was evaluated to determine whether "
            "sequentially correcting previous model errors could "
            "improve classification performance."
        ),
        "strength": (
            "Can model complex nonlinear relationships and often "
            "provides strong predictive performance on structured data."
        ),
        "limitation": (
            "More sensitive to parameter settings and can require "
            "careful tuning compared with simpler models."
        )
    }
}


def get_explanation(model_name):

    return MODEL_EXPLANATIONS.get(
        model_name,
        {}
    )


def compare_models(results):

    if results.empty:
        return None

    best = results.loc[
        results["F1 Score"].idxmax()
    ]

    worst = results.loc[
        results["F1 Score"].idxmin()
    ]

    return {
        "best_model": best["Model"],
        "best_f1": best["F1 Score"],
        "weakest_model": worst["Model"],
        "weakest_f1": worst["F1 Score"]
    }


def academic_summary(results):

    comparison = compare_models(results)

    if comparison is None:
        return ""

    best = comparison["best_model"]
    weakest = comparison["weakest_model"]

    return (
        f"The comparative evaluation identified {best} as the strongest "
        f"classifier according to weighted F1 score, while {weakest} "
        f"produced the lowest performance among the evaluated models. "
        "The models were evaluated using the same dataset preparation "
        "and evaluation criteria to provide a consistent comparison. "
        "The results support the use of the highest-performing model "
        "for the predictive stage while retaining the other models "
        "as comparative baselines."
    )