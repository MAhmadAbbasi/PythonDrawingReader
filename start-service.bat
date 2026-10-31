@echo off
rem Drawing2Quote: the drawing reader the POS calls (Sales > Drawing to Solution). Keep this window open.
cd /d "%~dp0"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
pause
