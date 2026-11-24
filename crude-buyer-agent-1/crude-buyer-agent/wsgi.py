"""Production entry point (Render): gunicorn wsgi:app --workers 1 --threads 8"""
import os
from app.server import create_app

if not os.getenv("DASHBOARD_TOKEN"):
    raise RuntimeError("Set DASHBOARD_TOKEN before deploying: the dashboard must not be public without a login token.")
app = create_app(start_scheduler=True)
