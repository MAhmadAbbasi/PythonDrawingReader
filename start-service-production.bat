@echo off
rem Drawing2Quote (production): run on the server behind the IIS site abc.asyntexconsultancy.com (deploy/web.config
rem passes the site's requests to http://127.0.0.1:8765). Settings: config/service.yaml, section production - set its
rem key to the one in the POS's appsettings.Production.json (DrawingService:Readers:python:Key).
rem Run it at start-up (Task Scheduler: "At system startup", or a service wrapper such as NSSM) so it stays up.
cd /d "%~dp0"
set D2Q_ENV=production
python -m app.serve
