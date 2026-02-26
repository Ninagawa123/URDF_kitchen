#!/usr/bin/env python3
"""
Minimal HTTP export service for URDF_kitchen headless batch export.

POST /export
{
  "inputPath": "/abs/path/to/source.urdf",
  "outputRoot": "/abs/path/to/output",
  "robotName": "my_robot"
}

Optional env:
- URDF_KITCHEN_SERVICE_HOST (default 0.0.0.0)
- URDF_KITCHEN_SERVICE_PORT (default 8091)
- URDF_KITCHEN_EXPORT_TIMEOUT_SECONDS (default 300)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORT_SCRIPT = REPO_ROOT / "scripts" / "headless_batch_export.py"

HOST = os.getenv("URDF_KITCHEN_SERVICE_HOST", "0.0.0.0")
PORT = int(os.getenv("URDF_KITCHEN_SERVICE_PORT", "8091"))
TIMEOUT_SECONDS = int(os.getenv("URDF_KITCHEN_EXPORT_TIMEOUT_SECONDS", "300"))


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/health":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return

        _json_response(
            self,
            200,
            {
                "ok": True,
                "service": "urdf-kitchen-export",
                "script": str(EXPORT_SCRIPT),
                "timeoutSeconds": TIMEOUT_SECONDS,
            },
        )

    def do_POST(self) -> None:
        if self.path != "/export":
            _json_response(self, 404, {"ok": False, "error": "not found"})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0:
            _json_response(self, 400, {"ok": False, "error": "empty request body"})
            return

        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception as exc:
            _json_response(self, 400, {"ok": False, "error": f"invalid JSON: {exc}"})
            return

        input_path = Path(str(payload.get("inputPath", "")).strip())
        output_root = Path(str(payload.get("outputRoot", "")).strip())
        robot_name = str(payload.get("robotName", "robot")).strip() or "robot"

        if not input_path.is_absolute() or not input_path.exists() or not input_path.is_file():
            _json_response(self, 400, {"ok": False, "error": f"invalid inputPath: {input_path}"})
            return

        if not output_root.is_absolute():
            _json_response(self, 400, {"ok": False, "error": f"outputRoot must be absolute: {output_root}"})
            return

        output_root.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(EXPORT_SCRIPT),
            "--input",
            str(input_path),
            "--out",
            str(output_root),
            "--robot-name",
            robot_name,
        ]

        env = os.environ.copy()
        env.setdefault("QT_QPA_PLATFORM", "offscreen")

        try:
            completed = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=TIMEOUT_SECONDS,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            _json_response(
                self,
                504,
                {
                    "ok": False,
                    "error": f"export timed out after {TIMEOUT_SECONDS}s",
                    "stdoutTail": _tail(exc.stdout or ""),
                    "stderrTail": _tail(exc.stderr or ""),
                },
            )
            return

        if completed.returncode != 0:
            _json_response(
                self,
                500,
                {
                    "ok": False,
                    "error": f"export failed (code={completed.returncode})",
                    "stdoutTail": _tail(completed.stdout),
                    "stderrTail": _tail(completed.stderr),
                },
            )
            return

        _json_response(
            self,
            200,
            {
                "ok": True,
                "inputPath": str(input_path),
                "outputRoot": str(output_root),
                "robotName": robot_name,
                "stdoutTail": _tail(completed.stdout),
                "stderrTail": _tail(completed.stderr),
            },
        )

    def log_message(self, format: str, *args: Any) -> None:
        return


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"URDF_kitchen export service listening on http://{HOST}:{PORT}")
    server.serve_forever()
