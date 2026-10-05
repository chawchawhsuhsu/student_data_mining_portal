
import pandas as pd


SKILL_COLUMNS = [
    "Vocabulary Mastery Confidence",
    "Grammar & Sentence Structure Confidence",
    "Listening Comprehension Confidence",
    "Speaking Fluency & Oral Expression Confidence",
    "Reading Confidence",
    "Writing Confidence"
]


def get_dataset_summary(df):
    return {
        "records": len(df),
        "variables": len(df.columns),
        "missing": int(df.isna().sum().sum()),
        "tracks": df["Observed Learning Progress Track"].nunique()
    }


def get_skill_averages(df):
    return df[SKILL_COLUMNS].mean()


def get_strongest_skill(profile):
    return max(SKILL_COLUMNS, key=lambda x: profile[x])


def get_weakest_skill(profile):
    return min(SKILL_COLUMNS, key=lambda x: profile[x])


def classify_learner(profile):
    study = profile["Daily Study Time (Minutes)"]
    speaking = profile["Speaking Fluency & Oral Expression Confidence"]
    grammar = profile["Grammar & Sentence Structure Confidence"]
    vocabulary = profile["Vocabulary Mastery Confidence"]

    if study < 20 and speaking < 3:
        return "Passive Study Trap Learner"

    if speaking >= 4 and profile["Daily English Exposure"] in [
        "High",
        "Very High"
    ]:
        return "Immersion-Driven Natural Communicator"

    if study >= 60 and speaking >= 4:
        return "High-Engagement Advanced Communicator"

    if study >= 45 and grammar >= 4:
        return "Structured Academic Learner"

    if profile["Daily English Exposure"] in ["Very Low", "Low"]:
        return "Low-Exposure / Casual Learner"

    if study >= 30 and grammar >= 3 and vocabulary >= 3:
        return "Balanced Consistent Learner"

    return "Developing Balanced Learner"

