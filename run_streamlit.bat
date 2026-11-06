@echo off
setlocal
python -m streamlit run app\app.py --server.address 127.0.0.1 --browser.gatherUsageStats false
endlocal
