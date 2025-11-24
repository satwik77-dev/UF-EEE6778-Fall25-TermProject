import streamlit as st
import time
import numpy as np

from InferencePipeline_Deliverable3 import (
    claimverify_infer
)


# ------------------------------------------------
# STREAMLIT PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="ClaimVerify",
    layout="wide",
    page_icon="🔍"
)

# ------------------------------------------------
# CUSTOM CSS (Dark Theme Styling)
# ------------------------------------------------
st.markdown("""
<style>

/* -------------------------
   GENERAL LAYOUT
-------------------------- */
body {
    color: #e6e6e6;
}

/* -------------------------
   INPUT BOX
-------------------------- */
textarea {
    background-color: #1e1e1e !important;
    color: #ffffff !important;
}

/* -------------------------
   BUTTON
-------------------------- */
.stButton button {
    background-color: #4da6ff;
    color: black;
    font-weight: 600;
    border-radius: 6px;
}

/* -------------------------
   VERDICT BOX
-------------------------- */
.verdict-box {
    padding: 18px;
    border-radius: 10px;
    color: white;
    font-size: 20px;
    font-weight: 600;
}

/* -------------------------
   EVIDENCE CARDS
-------------------------- */
.evidence-card {
    background-color: #1e1e1e;
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
    border: 1px solid #333;
    color: #e6e6e6;
}

.evidence-card a {
    color: #4da6ff !important;
    font-weight: bold;
}

/* -------------------------
   EXPLANATION TOKENS
-------------------------- */
.token {
    padding: 3px 5px;
    margin: 2px;
    border-radius: 4px;
    display: inline-block;
    color: black;
    font-size: 15px;
}

/* -------------------------
   FOOTER BOX
-------------------------- */
.footer-box {
    background-color: #003300;
    color: #aaffaa;
    padding: 12px;
    border-radius: 8px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)



# ------------------------------------------------
# PAGE HEADER
# ------------------------------------------------
st.title("🔍 ClaimVerify")
st.write("An AI-powered fact verification assistant with evidence, calibrated confidence, and explanations.")


# ------------------------------------------------
# USER INPUT
# ------------------------------------------------
st.subheader("Enter a claim to verify:")
user_claim = st.text_area("", height=120)

run_button = st.button("Check Claim")



# ------------------------------------------------
# RUN INFERENCE
# ------------------------------------------------
if run_button and user_claim.strip() != "":
    with st.spinner("Analyzing claim…"):
        result = claimverify_infer(user_claim)

    # --------------------------------------------
    # VERDICT DISPLAY
    # --------------------------------------------
    st.subheader("🎯 Verdict")

    verdict = result["verdict"]
    confidence = result["confidence"]

    # Color logic
    if verdict == "Likely False":
        color = "#e63946"
    elif verdict == "Likely True":
        color = "#2a9d8f"
    else:
        color = "#e9c46a"

    st.markdown(
        f"""
        <div class="verdict-box" style="background-color:{color};">
            {verdict} — Confidence: {confidence:.2f}
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------
    # EVIDENCE SECTION
    # --------------------------------------------
    st.subheader("📚 Retrieved Evidence")

    if result["evidence"]:
        for ev in result["evidence"]:
            st.markdown(
                f"""
                <div class="evidence-card">
                    <b>Rank {ev['rank']}</b><br>
                    <i>Claim:</i> {ev['claim_text']}<br>
                    <i>Source:</i> {ev['dataset_source']} — {ev['verdict_mapped']}<br>
                    <i>Similarity:</i> {ev['similarity']:.2f}<br><br>
                    <a href="{ev['url']}" target="_blank">View Source</a>
                </div>
                """,
                unsafe_allow_html=True
            )
    else:
        st.info("No evidence available.")


    # --------------------------------------------
    # EXPLANATION SECTION
    # --------------------------------------------
    st.subheader("🔍 Explanation (Token Importance)")

    if result["explanation"]:
        explanation_html = "<div>"

        for token_info in result["explanation"]:
            token = token_info["token"]
            score = token_info["score"]

            # convert score to intensity
            intensity = min(max(abs(score), 0.1), 0.9)

            if score >= 0:
                rgba = f"rgba(30, 144, 255, {intensity})"  # blue
            else:
                rgba = f"rgba(255, 69, 58, {intensity})"   # red

            explanation_html += (
                f'<span class="token" style="background-color:{rgba};">{token}</span>'
            )

        explanation_html += "</div>"

        st.markdown(explanation_html, unsafe_allow_html=True)
    else:
        st.info("No explanation available for this claim.")


    # --------------------------------------------
    # RUNTIME PROFILING
    # --------------------------------------------
    st.subheader("⏱ Runtime Profiling")
    st.json(result["runtime_ms"])


    # --------------------------------------------
    # SOURCE BLOCK
    # --------------------------------------------
    st.markdown(
        f"""
        <div class="footer-box">
            Retrieval Source: {result['source']}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.success("Inference complete.")
