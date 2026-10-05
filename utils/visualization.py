import plotly.express as px
import plotly.graph_objects as go

SKILLS = [
    "Vocabulary Mastery Confidence",
    "Grammar & Sentence Structure Confidence",
    "Listening Comprehension Confidence",
    "Speaking Fluency & Oral Expression Confidence",
    "Reading Confidence",
    "Writing Confidence",
]


def radar_chart(profile, averages):
  labels = ["Vocabulary", "Grammar", "Listening", "Speaking", "Reading", "Writing"]

  # Extract values safely
  learner_values = [profile.get(x, 0) for x in SKILLS]
  average_values = [averages.get(x, 0) for x in SKILLS]

  # Close the radar chart loop by repeating the first element
  learner_values.append(learner_values[0])
  average_values.append(average_values[0])
  labels.append(labels[0])

  fig = go.Figure()

  fig.add_trace(
      go.Scatterpolar(
          r=learner_values,
          theta=labels,
          fill="toself",
          name="Your Profile",
          fillcolor="rgba(31, 119, 180, 0.4)",  # Semi-transparent blue fill
          line=dict(color="#1f77b4"),
      )
  )

  fig.add_trace(
      go.Scatterpolar(
          r=average_values,
          theta=labels,
          fill="toself",
          name="Dataset Average",
          fillcolor="rgba(255, 127, 14, 0.3)",  # Semi-transparent orange fill
          line=dict(color="#ff7f0e"),
      )
  )

  fig.update_layout(
      polar=dict(radialaxis=dict(visible=True, range=[0, 5])),
      height=500,
      margin=dict(l=40, r=40, t=40, b=40),
  )

  return fig


def study_time_distribution(df):
  return px.histogram(
      df,
      x="Daily Study Time (Minutes)",
      title="Daily Study Time Distribution",
      nbins=20,
  )


def progress_distribution(df):
  return px.histogram(
      df,
      x="Observed Learning Progress Track",
      title="Learning Progress Distribution",
  )


def learning_approach_chart(df):
  return px.histogram(
      df,
      x="Primary Learning Approach",
      color="Observed Learning Progress Track",
      barmode="group",
      title="Learning Approach vs Progress",
  )