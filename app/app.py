"""Local-first Streamlit dashboard for casting defect inspection."""
from __future__ import annotations

import hashlib
import html
import tempfile
from io import BytesIO
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from src.inference.app_storage import add_inspection, clear_inspections, get_last_30_minutes, get_recent_inspections
from src.inference.input_gate import check_input
from src.inference.predict import ModelUnavailableError, predict

MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(page_title="Casting Inspection AI", page_icon=None, layout="wide")
st.markdown("""
<style>
html,body,[data-testid="stAppViewContainer"]{background:#eef2f5!important;color:#101820!important}
[data-testid="stHeader"]{background:transparent!important}
.block-container{max-width:1780px!important;padding:16px 20px 12px!important}
.dashboard-header{min-height:72px;background:#101820;border-radius:9px;display:flex;align-items:center;justify-content:space-between;padding:0 20px;margin-bottom:13px}
.title{color:#fff;font-size:24px;font-weight:900;letter-spacing:1.4px}
.subtitle{color:#aebbc4;font-size:9px;letter-spacing:1.8px;margin-top:6px;font-weight:700}
.result{min-height:126px;border-radius:8px;border:2px solid;display:flex;align-items:center;justify-content:center;flex-direction:column}
.pass{background:#eaf8f0;border-color:#18a957}.fail{background:#fff0f1;border-color:#dc3341}
.review{background:#fff7e3;border-color:#d98b00}.invalid{background:#fff0f1;border-color:#dc3341}
.result-text{color:#101820;font-size:42px;font-weight:950;letter-spacing:2.5px}
.footer{text-align:center;color:#72808a;font-size:9px;padding-top:10px}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dashboard-header">
  <div><div class="title">CASTING INSPECTION AI</div>
  <div class="subtitle">AI-BASED CASTING QUALITY INSPECTION SYSTEM</div></div>
  <div style="color:#dce5ea;font-size:10px;font-weight:800">LOCAL INSPECTION WORKSTATION</div>
</div>
""", unsafe_allow_html=True)

upload_col, camera_col, result_col = st.columns([1.05, 1.05, 0.90], gap="small")
with upload_col:
    st.markdown("**UPLOAD / DRAG & DROP**")
    uploaded_file = st.file_uploader("Upload casting image", type=["jpg", "jpeg", "png"])
with camera_col:
    st.markdown("**LIVE CAMERA**")
    camera_file = st.camera_input("Capture inspection photo")
input_file = camera_file if camera_file is not None else uploaded_file

result = st.session_state.get("inspection_result")
image_name = None
image_bytes = None
if input_file is not None:
    image_name = "camera_capture.jpg" if camera_file is not None else "uploaded_image.jpg"
    image_bytes = input_file.getvalue()
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        st.error("Image exceeds the 10 MB upload limit.")
    elif st.session_state.get("last_inspected_hash") != image_hash:
        try:
            suffix = ".jpg" if camera_file is not None else (Path(image_name).suffix.lower() or ".jpg")
            with tempfile.TemporaryDirectory(prefix="casting-inspection-") as temp_dir:
                temp_path = Path(temp_dir) / f"inspection{suffix}"
                temp_path.write_bytes(image_bytes)
                gate = check_input(temp_path)
                if not gate["valid"]:
                    result = {"result": "INVALID", "defect": "not_a_casting", "confidence": 0.0, "reason": gate["reason"]}
                else:
                    with st.spinner("AI INSPECTION IN PROGRESS..."):
                        result = predict(temp_path)
            add_inspection(image_name, str(result["result"]), str(result["defect"]), float(result["confidence"]))
            st.session_state["inspection_result"] = result
            st.session_state["last_inspected_hash"] = image_hash
        except (ModelUnavailableError, FileNotFoundError, ValueError, RuntimeError) as exc:
            result = {"result": "ERROR", "message": str(exc)}
            st.session_state["inspection_result"] = result
            st.session_state["last_inspected_hash"] = image_hash
        except Exception as exc:
            result = {"result": "ERROR", "message": f"Inspection could not be completed: {type(exc).__name__}"}
            st.session_state["inspection_result"] = result
            st.session_state["last_inspected_hash"] = image_hash

with result_col:
    st.markdown("**INSPECTION RESULT**")
    if result is None:
        st.info("READY — Awaiting part image")
    elif result.get("result") == "ERROR":
        st.error("ERROR — Inspection not completed")
        st.error(result.get("message", "Unknown inspection error."))
    elif result.get("result") == "INVALID":
        st.warning("INVALID — Input rejected")
        st.warning(result.get("reason", "Image did not pass input validation."))
    else:
        status = str(result["result"])
        st.markdown(f'<div class="result {status.lower()}"><div class="result-text">{html.escape(status)}</div></div>', unsafe_allow_html=True)
        metric_a, metric_b = st.columns(2)
        metric_a.metric("Defect type", str(result["defect"]).replace("_", " ").upper())
        metric_b.metric("Confidence", f'{float(result["confidence"]):.1%}')

st.divider()
history_col, image_col, chart_col = st.columns([0.92, 1.12, 1.80], gap="small")
with history_col:
    st.markdown("**HISTORY — LAST 15 PARTS**")
    if st.button("CLEAR LOCAL HISTORY", use_container_width=True):
        clear_inspections()
        st.session_state.pop("inspection_result", None)
        st.session_state.pop("last_inspected_hash", None)
        st.rerun()
    history = get_recent_inspections(limit=15)
    if history:
        table = pd.DataFrame(history).rename(columns={
            "timestamp":"Time", "image_name":"Image", "result":"Result",
            "defect_type":"Defect", "confidence":"Confidence",
        })
        table["Confidence"] = table["Confidence"].map(lambda value: f"{float(value):.1%}")
        st.dataframe(table[["Time","Image","Result","Defect","Confidence"]], hide_index=True, use_container_width=True)
    else:
        st.info("No inspections recorded on this device.")
with image_col:
    st.markdown("**PART IMAGE**")
    if image_bytes:
        try:
            with Image.open(BytesIO(image_bytes)) as source:
                st.image(source.convert("RGB"), use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError):
            st.warning("Image preview could not be decoded.")
    else:
        st.info("No image selected.")
with chart_col:
    st.markdown("**DEFECT DISTRIBUTION — LAST 30 MINUTES**")
    rows = [
        {"Class":item["defect_type"].replace("_"," ").upper(),"Result":item["result"]}
        for item in get_last_30_minutes() if item["result"] in {"PASS","FAIL","REVIEW"}
    ]
    if rows:
        summary = pd.DataFrame(rows).groupby(["Class","Result"],as_index=False).size()
        chart = alt.Chart(summary).mark_bar().encode(
            y=alt.Y("Class:N",sort="-x",title=None),
            x=alt.X("size:Q",title="Inspections"),
            color=alt.Color("Result:N",title="Decision"),
            tooltip=["Class","Result","size"],
        ).properties(height=340)
        st.altair_chart(chart,use_container_width=True)
    else:
        st.info("Chart appears after inspections are recorded.")

st.markdown('<div class="footer">RESNET18 · 12 CLASSES · TEMPORARY IMAGE PROCESSING · LOCAL METADATA HISTORY</div>',unsafe_allow_html=True)
with st.expander("Privacy and model notes"):
    st.write("Images are written to a temporary local file only during inspection and removed afterwards. Local SQLite stores timestamp, a generic source label (not the user-supplied filename), decision, predicted class, and confidence; it does not store image bytes. The application does not call an external prediction API.")
    st.write("The face gate is a basic frontal-face exclusion heuristic, not a casting-object detector. Confidence below 70% becomes REVIEW. This prototype is not certified for final production acceptance.")
