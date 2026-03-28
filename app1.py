# app.py

import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Placement Predictor", layout="wide")

# -----------------------------
# CSS (Fix visibility)
# -----------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #141E30, #243B55);
    color: white;
}

label, .stMarkdown, .stText, .stTitle, .stSubheader {
    color: white !important;
}

div[data-testid="stWidgetLabel"] {
    color: white !important;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 style='text-align:center;'>🎓 Placement Prediction Dashboard</h1>", unsafe_allow_html=True)

# -----------------------------
# Load Models
# -----------------------------
model = pickle.load(open("artifacts/model.pkl", "rb"))
scaler = pickle.load(open("artifacts/scaler.pkl", "rb"))
encoders = pickle.load(open("artifacts/encoders.pkl", "rb"))

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([2, 1])

# -----------------------------
# INPUT SECTION
# -----------------------------
with left:
    st.subheader("📊 Student Profile")

    col1, col2, col3 = st.columns(3)

    with col1:
        IQ = st.number_input("IQ", 50, 200, 100)
        cgpa = st.number_input("CGPA", 0.0, 10.0, 6.5)

    with col2:
        prev_sem = st.number_input("Previous Semester", 0.0, 10.0, 6.0)
        communication = st.slider("Communication Skills", 1, 10, 5)

    with col3:
        internship = st.selectbox("Internship", ["Yes", "No"])
        projects = st.number_input("Projects", 0, 10, 2)

    academic_perf = st.slider("Academic Performance", 1, 10, 5)
    extra_curricular = st.slider("Extra Curricular", 1, 10, 5)

    predict_btn = st.button("🚀 Predict Placement")

# -----------------------------
# RESULT SECTION
# -----------------------------
with right:
    st.subheader("📈 Prediction Result")

    if not predict_btn:
        st.info("👉 Enter details and click 'Predict Placement' to see results.")

    else:
        input_data = [
            IQ,
            prev_sem,
            cgpa,
            academic_perf,
            encoders["Internship_Experience"].transform([internship])[0],
            extra_curricular,
            communication,
            projects
        ]

        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = model.predict(input_scaled)[0]

        # -----------------------------
        # RESULT MESSAGE
        # -----------------------------
        if prediction == 1:
            st.balloons()
            st.success("🎉 Congratulations! You have high chances of getting placed!")
            st.markdown("### 🏆 Keep up the great work! You're on the right track.")

        else:
            st.error("⚠️ Your chances of placement are currently low.")
            st.markdown("### 💪 Don't worry! Follow the suggestions below to improve.")

        # -----------------------------
        # PROFILE STRENGTH (Radar Chart)
        # -----------------------------
        st.markdown("### 📈 Profile Strength")

        labels = ["CGPA", "Internship", "Communication", "Projects", "Academics"]

        values = [
            cgpa/10,
            1 if internship == "Yes" else 0,
            communication/10,
            projects/5,
            academic_perf/10
        ]

        values += values[:1]

        angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
        angles += angles[:1]

        fig2, ax2 = plt.subplots(subplot_kw=dict(polar=True))

        ax2.plot(angles, values)
        ax2.fill(angles, values, alpha=0.3)

        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(labels, color="white")

        ax2.set_facecolor("#243B55")
        fig2.patch.set_facecolor("#243B55")

        st.pyplot(fig2)

        # -----------------------------
        # SMART SUGGESTIONS
        # -----------------------------
        st.markdown("### 💡 Suggestions to Improve Placement Chances")

        tips = []

        if cgpa < 7:
            tips.append("📚 Improve your CGPA to above 7")

        if internship == "No":
            tips.append("💼 Gain internship experience")

        if communication < 6:
            tips.append("🗣️ Work on communication skills")

        if projects < 2:
            tips.append("💻 Build at least 2 real-world projects")

        if extra_curricular < 5:
            tips.append("🎯 Participate in extracurricular activities")

        # -----------------------------
        # DISPLAY LOGIC
        # -----------------------------
        if prediction == 1:
            if tips:
                st.info("✨ You are doing well, but you can improve further:")
                for tip in tips:
                    st.warning(tip)
            else:
                st.success("🔥 Excellent profile! You are placement-ready!")

        else:
            if tips:
                for tip in tips:
                    st.warning(tip)
            else:
                st.info("👍 Improve consistency and keep practicing.")

# -----------------------------
# BOTTOM SECTION
# -----------------------------
st.markdown("---")

colA, colB = st.columns(2)

with colA:
    st.subheader("📘 Placement Guidelines")
    st.markdown("""
    - Maintain CGPA above 7  
    - Gain internship experience  
    - Improve communication skills  
    - Build strong projects  
    """)

with colB:
    st.subheader("🚀 Pro Tips")
    st.markdown("""
    - Practice coding daily  
    - Build GitHub portfolio  
    - Prepare for interviews  
    - Learn industry skills  
    """)

# -----------------------------
# USER GUIDE
# -----------------------------
st.markdown("---")
st.header("📘 User Guide")

tab1, tab2, tab3 = st.tabs(["Input Guide", "Placement Strategy", "Career Growth"])

with tab1:
    st.write("Fill student academic and skill details accurately.")

with tab2:
    st.write("Focus on CGPA, internships, and projects.")

with tab3:
    st.write("Practice coding, build projects, and prepare interviews.")