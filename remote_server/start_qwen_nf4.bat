@echo off
echo ============================================
echo   Starting Qwen Encoder Server (NF4)
echo   Conda env: remote-qwen
echo   Server: remote_text_encoder_server_NF4.py
echo ============================================

REM === Activate Anaconda ===
CALL D:\anaconda3\Scripts\activate.bat D:\anaconda3

REM === Activate conda environment ===
CALL conda activate remote-qwen

REM === Switch to project directory ===
cd /d D:\Models\remote_qwen_server

REM === Optional: enable token debug ===
set QWEN_DEBUG_TOKENS=1

REM === Launch server ===
python remote_text_encoder_server_NF4.py

pause
