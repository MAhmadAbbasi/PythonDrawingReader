"""Runs the service with its environment's settings (config/service.yaml):
  python -m app.serve                        development (D2Q_ENV unset)
  set D2Q_ENV=production & python -m app.serve"""
import uvicorn

from .settings import SETTINGS

if __name__ == "__main__":
    print(f"Drawing2Quote (Python) - {SETTINGS['environment']} on http://{SETTINGS['host']}:{SETTINGS['port']}; "
          f"service key {'required' if SETTINGS['key'] else 'not required'}", flush=True)
    uvicorn.run("app.main:app", host=SETTINGS["host"], port=SETTINGS["port"], proxy_headers=True)
