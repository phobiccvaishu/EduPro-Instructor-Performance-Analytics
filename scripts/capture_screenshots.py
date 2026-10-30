"""
scripts/capture_screenshots.py
==============================
Automated screenshot generator for EduPro Analytics Platform.
Spawns Streamlit, launches headless Chrome with remote debugging,
and executes CDP capture via Node.js to wait for Streamlit hydration.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CHROME_PATHS = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def find_browser() -> Path:
    for p in CHROME_PATHS:
        if p.exists():
            return p
    raise FileNotFoundError("Neither Microsoft Edge nor Google Chrome was found.")


def wait_for_url(url: str, timeout: float = 30.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=2) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main():
    repo_root = Path(__file__).resolve().parent.parent
    docs_dir = repo_root / "docs" / "screenshots"
    docs_dir.mkdir(parents=True, exist_ok=True)

    browser = find_browser()
    st_port = 8533
    cdp_port = 9222
    st_base = f"http://localhost:{st_port}"
    cdp_base = f"http://127.0.0.1:{cdp_port}/json/version"

    user_data_dir = tempfile.mkdtemp(prefix="chrome_cdp_")

    st_cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(repo_root / "app.py"),
        "--server.port",
        str(st_port),
        "--server.headless",
        "true",
        "--server.fileWatcherType",
        "none",
        "--browser.serverAddress",
        "localhost",
    ]

    chrome_cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        f"--remote-debugging-port={cdp_port}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
    ]

    print("Starting Streamlit server...")
    st_proc = subprocess.Popen(st_cmd, cwd=str(repo_root))

    print("Starting headless browser with CDP...")
    chrome_proc = subprocess.Popen(chrome_cmd)

    try:
        print("Waiting for Streamlit server...")
        if not wait_for_url(f"{st_base}/_stcore/health", timeout=30):
            raise RuntimeError("Streamlit server failed to start.")

        print("Waiting for Browser CDP...")
        if not wait_for_url(cdp_base, timeout=30):
            raise RuntimeError("Chrome CDP failed to start.")

        print("Running CDP capture script...")
        node_script = repo_root / "scripts" / "capture_cdp.mjs"
        res = subprocess.run(
            ["node", str(node_script)], cwd=str(repo_root), capture_output=True, text=True
        )
        print("Node output:\n", res.stdout)
        if res.returncode != 0:
            print("Node stderr:\n", res.stderr)
            raise RuntimeError(f"CDP capture failed with exit code {res.returncode}")

        print("Capture completed successfully.")

    finally:
        print("Cleaning up processes...")
        chrome_proc.terminate()
        st_proc.terminate()
        try:
            chrome_proc.wait(timeout=5)
        except Exception:
            chrome_proc.kill()
        try:
            st_proc.wait(timeout=5)
        except Exception:
            st_proc.kill()
        shutil.rmtree(user_data_dir, ignore_errors=True)
        print("Done.")


if __name__ == "__main__":
    main()
