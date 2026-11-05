from __future__ import annotations

from datetime import datetime

import streamlit as st

from src.inference import (
    DEFAULT_CHECKPOINT,
    ModelUnavailableError,
    load_model,
    open_image_file,
    predict_image,
)

st.set_page_config(
    page_title="Casting Defect AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1180px;}
      .hero {padding: 1.35rem 1.5rem; border: 1px solid rgba(128,128,128,.28);
             border-radius: 14px; margin-bottom: 1.2rem;}
      .muted {color: #808080; font-size: .92rem;}
      div[data-testid="stMetric"] {border: 1px solid rgba(128,128,128,.25);
             padding: 1rem; border-radius: 12px;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Casting Defect AI")
st.caption("AI-assisted visual inspection for cast components")

st.markdown(
    """
    <div class="hero">
      <strong>Quality inspection workstation</strong><br>
      Upload a component image, run the trained classifier, and review the
      predicted inspection result.
      <div class="muted">Demonstration software — verify every result using your approved quality process.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource(show_spinner="Loading trained inspection model...")
def get_model():
    return load_model(DEFAULT_CHECKPOINT)

left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("1. Component image")
    uploaded = st.file_uploader(
        "Upload a JPG, JPEG, or PNG image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear image of one casting component.",
    )
    if uploaded:
        try:
            preview = open_image_file(uploaded)
            st.image(preview, caption="Image selected for inspection", use_container_width=True)
        except ValueError as exc:
            st.error(str(exc))
            preview = None
    else:
        preview = None
        st.info("Choose an image to begin inspection.")

with right:
    st.subheader("2. Inspection result")
    inspect_clicked = st.button(
        "Run inspection",
        type="primary",
        use_container_width=True,
        disabled=preview is None,
    )

    if inspect_clicked and preview is not None:
        try:
            model, device = get_model()
            with st.spinner("Analysing component image..."):
                result = predict_image(preview, model, device)

            st.session_state["last_result"] = {
                **result,
                "timestamp": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                "filename": uploaded.name,
            }
        except ModelUnavailableError as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Inspection could not be completed: {exc}")

    result = st.session_state.get("last_result")
    if result:
        st.divider()
        if result["decision"] == "PASS":
            st.success("PASS — No defect class was predicted.")
        else:
            st.error("FAIL — A defect class was predicted.")

        m1, m2 = st.columns(2)
        m1.metric("Predicted category", result["defect"])
        m2.metric("Model confidence", f"{result['confidence'] * 100:.1f}%")
        st.caption(f"File: {result['filename']}  |  Inspected: {result['timestamp']}")
        with st.expander("Technical result"):
            st.json(result)
    else:
        st.info("Your inspection result will appear here.")

st.divider()
with st.expander("Model and inspection notes"):
    st.write(f"Expected checkpoint: `{DEFAULT_CHECKPOINT}`")
    st.write(
        "The configured inference pipeline expects a 12-class ResNet18 state dictionary. "
        "Its image size, normalization, and class order must match the original training setup."
    )
    st.warning(
        "A missing or incompatible checkpoint will stop inference. The app does not generate "
        "fake predictions or treat an unknown result as PASS."
    )

st.caption("Casting Defect AI • Prototype for demonstration and evaluation")
