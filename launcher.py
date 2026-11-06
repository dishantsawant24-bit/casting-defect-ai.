"""Windows launcher for the local Streamlit inspection application."""
from __future__ import annotations

import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


def bundle_root() -> Path:
    if getattr(sys, "frozen", False):
        executable_dir = Path(sys.executable).resolve().parent
        internal_dir = executable_dir / "_internal"
        return internal_dir if internal_dir.is_dir() else executable_dir
    return Path(__file__).resolve().parent


def main() -> None:
    root = bundle_root()
    app_file = root / "app" / "app.py"
    if not app_file.is_file():
        raise FileNotFoundError("Bundled application entry point is missing.")
    os.chdir(root)
    local_app_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local"))
    os.environ.setdefault("CASTING_INSPECTION_APP_DATA", str(local_app_data / "CastingInspectionAI"))
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = int(sock.getsockname()[1])
    options = {
        "server.port": port,
        "server.address": "127.0.0.1",
        "server.headless": True,
        "browser.gatherUsageStats": False,
        "server.fileWatcherType": "none",
    }

    from streamlit import config
    from streamlit.web import bootstrap

    for key, value in options.items():
        config.set_option(key, value)

    def open_browser() -> None:
        time.sleep(2.0)
        webbrowser.open(f"http://127.0.0.1:{port}")

    threading.Thread(target=open_browser, daemon=True).start()
    bootstrap.run(str(app_file), False, [], options)


if __name__ == "__main__":
    main()
