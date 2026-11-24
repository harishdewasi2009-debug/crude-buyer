"""Start the dashboard:  python run.py   ->  http://127.0.0.1:8000"""
import os
from app.server import create_app

if __name__ == "__main__":
    app = create_app(start_scheduler=True)
    host = os.getenv("CBA_HOST", "127.0.0.1")  # keep local unless you set DASHBOARD_TOKEN and put it behind HTTPS
    app.run(host=host, port=int(os.getenv("CBA_PORT", "8000")), threaded=True)
