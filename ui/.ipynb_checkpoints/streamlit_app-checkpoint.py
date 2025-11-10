# ClaimVerify: AI-Powered Fact Verification System

# ----------------------------
# TEMPORARY DEMO PIPELINE MOCK
# ----------------------------

def claimverify_infer(user_claim: str):
    """
    Dummy fallback inference function for demo purposes.
    Always returns a simulated response without model execution.
    """

    # Example simulated result
    sample_explanation = [
        {"token": "COVID-19", "score": 0.9},
        {"token": "vaccines", "score": -0.5},
        {"token": "cause", "score": 0.8},
        {"token": "infertility", "score": -0.3},
    ]

    sample_evidence = [
        {
            "claim_text": "COVID-19 vaccines cause infertility.",
            "similarity": 0.78,
            "verdict": "Likely False",
            "url": "https://www.politifact.com/factchecks/2021/may/01/fact-check/",
            "dataset_source": "PolitiFact"
        },
        {
            "claim_text": "COVID-19 vaccines do not affect fertility according to studies.",
            "similarity": 0.74,
            "verdict": "Likely True",
            "url": "https://www.snopes.com/fact-check/vaccine-fertility/",
            "dataset_source": "Snopes"
        }
    ]

    return {
        "verdict": "Likely False",
        "confidence": 0.86,
        "explanation": sample_explanation,
        "evidence": sample_evidence,
        "source": "Demo Mode"
    }

#Importing Required Libraries
import streamlit as st
from pathlib import Path
import sys
import os
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.inference_pipeline import claimverify_infer


# App Configuration
st.set_page_config(
    page_title="ClaimVerify: AI-Powered Fact Verification System",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Paths
BASE_PATH = Path(__file__).resolve().parents[1]
ASSETS_PATH = BASE_PATH / "ui" / "assets"


# Theme Control (Dark / Light)
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"

def set_theme():
    """Inject CSS depending on selected theme."""
    theme_file = ASSETS_PATH / f"{st.session_state['theme']}_theme.css"
    with open(theme_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Sidebar for Theme Toggle
with st.sidebar:
    st.header("Settings")
    theme_choice = st.radio("Choose Theme:", ["Dark", "Light"], index=0 if st.session_state["theme"] == "dark" else 1)
    st.session_state["theme"] = theme_choice.lower()

# Apply Theme
set_theme()


# Header Section
st.markdown("""
<div class="header-container">
    <h1>ClaimVerify: AI-Powered Fact Verification System</h1>
    <p>Verify factual claims with AI-powered retrieval, classification, and transparent explanations.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# ================================================
# Input Section
# ================================================
st.subheader("Enter a claim to fact-check:")
user_claim = st.text_area(
    label="",
    placeholder="e.g. The unemployment rate reached a 50-year low in 2024.",
    height=100
)

run_check = st.button("Check This Claim")

# ================================================
# Inference + Results
# ================================================
if run_check and user_claim.strip():
    with st.spinner("Analyzing claim with ClaimVerify..."):
        result = claimverify_infer(user_claim)

    # Verdict Colors
    verdict = result["verdict"]
    conf = result["confidence"]
    color_map = {
        "Likely True": "#4CAF50",    # Green
        "Likely False": "#F44336",   # Red
        "Uncertain": "#FF9800"       # Orange
    }
    verdict_color = color_map.get(verdict, "#6C63FF")

    # Verdict Section
    st.markdown(f"""
    <div class="verdict-card" style="background-color: {verdict_color};">
        <h2>VERDICT: {verdict}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.progress(conf)
    st.write(f"**Confidence:** {conf*100:.1f}%")
    st.caption(f"Source: {result['source']}")

    st.divider()

    # Explanation Section
    st.subheader("Why This Verdict?")
    claim_tokens = result["explanation"]

    # Generate HTML highlight for tokens
    html_text = ""
    for tok in claim_tokens:
        score = tok["score"]
        color = "rgba(76, 175, 80, {:.2f})".format(abs(score)) if score > 0 else "rgba(244, 67, 54, {:.2f})".format(abs(score))
        html_text += f"<span class='token-highlight' style='background-color: {color}; padding: 2px 4px; margin: 1px; border-radius: 4px;'>{tok['token']}</span> "

    st.markdown(f"""
    <div class="explanation-box">
        {html_text}
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Explanation Legend"):
        st.markdown("""
        - <span style='color:#4CAF50;'>Green</span>: Supports claim  
        - <span style='color:#F44336;'>Red</span>: Contradicts claim
        """, unsafe_allow_html=True)

    st.divider()

    # Evidence Section
    st.subheader("Supporting Evidence")
    for ev in result["evidence"]:
        verdict_badge = ev["verdict"].upper()
        badge_color = "#4CAF50" if "True" in verdict_badge else "#F44336" if "False" in verdict_badge else "#FF9800"
        st.markdown(f"""
        <div class="evidence-card">
            <div class="evidence-header">
                <span class="badge" style="background-color:{badge_color};">{verdict_badge}</span>
                <span class="source">{ev['dataset_source']}</span>
            </div>
            <div class="evidence-body">
                <p>{ev['claim_text']}</p>
                <a href="{ev['url']}" target="_blank">Read full fact-check →</a>
            </div>
            <div class="evidence-footer">
                Similarity Score: {ev['similarity']*100:.1f}%
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Feedback Section
    st.subheader("Was this fact-check helpful?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("Yes, Accurate")
    with col2:
        st.button("No, Inaccurate")
    with col3:
        st.button("Provide Feedback")

else:
    st.info("Enter a claim above and click 'Check This Claim' to begin analysis.")
