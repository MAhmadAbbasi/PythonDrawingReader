@echo off
rem Drawing2Quote (development): the drawing reader the POS calls (Sales > Drawing to Solution) on
rem http://127.0.0.1:8765 - the POS's appsettings.Development.json, DrawingService:Readers:python. Keep this window open.
cd /d "%~dp0"
set D2Q_ENV=development
python -m app.serve
pause
