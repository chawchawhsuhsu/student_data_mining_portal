import streamlit as st

# ============================================================
# PAGE CONFIGURATION (Must be the first Streamlit command)
# ============================================================
st.set_page_config(
    page_title="English Learning Data Mining Portal",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# MAIN ENTRYWAY / NAVIGATION LANDING
# ============================================================

st.title("📚 English Learning Data Mining Portal")
st.caption("FP-Growth Association Rules & Machine Learning Classification")

st.markdown("---")

st.markdown("""
### Welcome to the Data Mining Portal!

Select a page from the **sidebar menu** to explore different sections of the application:

1. **Overview**: Project background, dataset summary, and methodology rationale (Why FP-Growth & Decision Tree).
2. **Dataset Analytics**: Interactive data distributions, histograms, and visual patterns ported from Colab.
3. **Descriptive Analytics**: Enter your profile to see your assigned Learner Persona and skill benchmark charts.
4. **Mining Logs**: Filter and search through mined FP-Growth association rules (Support, Confidence, Lift).
5. **Predictive Engine**: Input learner parameters to run predictions and get actionable improvement recommendations.
""")

st.info("👈 Use the left sidebar to navigate between pages.")

st.markdown("---")

# Safe HTML Footer Rendering
st.markdown(
    """
    <div style="text-align: center; padding: 1rem; color: #64748b; font-size: 0.85rem;">
        English Learning Data Mining Portal | Academic Analytics & Educational AI Demonstration
    </div>
    """,
    unsafe_allow_html=True
)