"""Minimal web UI for orchestrating the AIR-BENCH setup and update pipelines.

This server is a thin driver around the existing CLIs (``pipeline.py`` and
``setup.py``). It spawns them as subprocesses, streams their output to the
browser, detects the ``review_json`` human-in-the-loop pause, serves the
referenced JSON checkpoint to an in-browser editor, and resumes the run by
writing a newline to the subprocess's stdin once edits are saved.

Nothing in pipeline.py / setup.py is modified or imported; the entire existing
CLI flag surface is reused by passing flags straight through.
"""

import json
import os
import subprocess
import sys
import threading
from pathlib import Path

from flask import Flask, jsonify, request, send_file

ROOT = Path(__file__).resolve().parent.parent
VENV_PYTHON = ROOT / "venv" / "bin" / "python"
PYTHON = str(VENV_PYTHON if VENV_PYTHON.exists() else sys.executable)
INDEX_HTML = Path(__file__).resolve().parent / "index.html"

# Tool name -> script path. Both scripts share pipeline.review_json, so the same
# pause/resume protocol drives either one.
TOOLS = {
    "pipeline": ROOT / "pipeline.py",
    "setup": ROOT / "setup.py",
}

# Markers emitted by pipeline.review_json (kept in sync with that function).
EDIT_MARKER = "[review] Edit this file as needed: "
MSG_MARKER = "[review] "
PAUSE_MARKER = "Press Enter after saving your edits to continue."

MAX_LOG_LINES = 5000

app = Flask(__name__, static_folder=None)


class Runner:
    """Owns the single active pipeline/setup subprocess and its parsed state."""

    def __init__(self):
        self.proc = None
        self.lock = threading.Lock()
        self._reset()

    def _reset(self):
        self.logs = []
        self.state = "idle"  # idle | running | paused | done | error
        self.review = None  # {"path": str, "message": str} when paused
        self.run_dir = None
        self.tool = None
        self.exit_code = None
        self._last_edit_path = None
        self._last_message = None

    # ---- lifecycle -------------------------------------------------------
    def start(self, tool, args):
        with self.lock:
            if self.state in ("running", "paused"):
                raise RuntimeError("A run is already active. Stop it first.")
            script = TOOLS[tool]
            self._reset()
            self.state = "running"
            self.tool = tool
            # -u keeps stdout unbuffered so prints and the input() prompt flush
            # immediately. --yes is intentionally never added: the pause is how
            # the web UI gets a chance to edit.
            cmd = [PYTHON, "-u", str(script), *args]
            self.logs.append("$ " + " ".join(cmd))
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )
        threading.Thread(target=self._reader, daemon=True).start()

    def resume(self):
        with self.lock:
            if self.state != "paused" or not self.proc:
                raise RuntimeError("Run is not paused.")
            self.proc.stdin.write(b"\n")
            self.proc.stdin.flush()
            self.state = "running"
            self.review = None

    def stop(self):
        with self.lock:
            if self.proc and self.proc.poll() is None:
                self.proc.terminate()
            self.state = "idle"
            self.review = None

    # ---- output parsing --------------------------------------------------
    def _emit_line(self, line):
        with self.lock:
            self.logs.append(line)
            if len(self.logs) > MAX_LOG_LINES:
                del self.logs[: len(self.logs) - MAX_LOG_LINES]
        if line.startswith(EDIT_MARKER):
            path = line[len(EDIT_MARKER):].strip()
            self._last_edit_path = path
            try:
                self.run_dir = str(Path(path).parent)
            except (ValueError, OSError):
                pass
        elif line.startswith(MSG_MARKER):
            msg = line[len(MSG_MARKER):].strip()
            if not msg.startswith("Edit this file") and PAUSE_MARKER not in msg:
                self._last_message = msg

    def _reader(self):
        fd = self.proc.stdout.fileno()
        buf = ""
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                break
            buf += chunk.decode("utf-8", errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                self._emit_line(line)
            # input() prints its prompt without a trailing newline, so the pause
            # is detected on the still-pending buffer.
            if buf.endswith(PAUSE_MARKER):
                self._emit_line(buf)
                buf = ""
                with self.lock:
                    self.state = "paused"
                    self.review = {
                        "path": self._last_edit_path,
                        "message": self._last_message,
                    }
        if buf:
            self._emit_line(buf)
        self.proc.wait()
        with self.lock:
            self.exit_code = self.proc.returncode
            self.state = "done" if self.proc.returncode == 0 else "error"
            self.review = None

    def status(self):
        with self.lock:
            files = []
            if self.run_dir and Path(self.run_dir).is_dir():
                files = sorted(p.name for p in Path(self.run_dir).glob("*.json"))
            return {
                "state": self.state,
                "tool": self.tool,
                "logs": list(self.logs),
                "review": dict(self.review) if self.review else None,
                "run_dir": self.run_dir,
                "files": files,
                "exit_code": self.exit_code,
            }


runner = Runner()


def safe_path(raw):
    """Resolve a path and ensure it stays within the repo root."""
    p = Path(raw).resolve()
    root = ROOT.resolve()
    if p != root and root not in p.parents:
        raise ValueError("Path is outside the project root.")
    return p


# ---- routes --------------------------------------------------------------
@app.get("/")
def index():
    return send_file(INDEX_HTML)


@app.get("/api/tools")
def api_tools():
    return jsonify(tools=sorted(TOOLS))


@app.post("/api/start")
def api_start():
    data = request.get_json(force=True) or {}
    tool = data.get("tool")
    args = data.get("args", [])
    if tool not in TOOLS:
        return jsonify(error=f"Unknown tool: {tool!r}"), 400
    if not isinstance(args, list) or not all(isinstance(a, str) for a in args):
        return jsonify(error="args must be a list of strings."), 400
    try:
        runner.start(tool, args)
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409
    return jsonify(ok=True)


@app.get("/api/status")
def api_status():
    return jsonify(runner.status())


@app.get("/api/file")
def api_get_file():
    raw = request.args.get("path", "")
    try:
        p = safe_path(raw)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    if not p.is_file():
        return jsonify(error="File not found."), 404
    return jsonify(path=str(p), content=p.read_text(encoding="utf-8"))


def _save(raw, content):
    p = safe_path(raw)
    try:
        json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    p.write_text(content, encoding="utf-8")
    return p


@app.post("/api/file")
def api_save_file():
    data = request.get_json(force=True) or {}
    try:
        _save(data.get("path", ""), data.get("content", ""))
    except ValueError as exc:
        return jsonify(error=str(exc)), 400
    return jsonify(ok=True)


@app.post("/api/continue")
def api_continue():
    data = request.get_json(silent=True) or {}
    if "path" in data and "content" in data:
        try:
            _save(data["path"], data["content"])
        except ValueError as exc:
            return jsonify(error=str(exc)), 400
    try:
        runner.resume()
    except RuntimeError as exc:
        return jsonify(error=str(exc)), 409
    return jsonify(ok=True)


@app.post("/api/stop")
def api_stop():
    runner.stop()
    return jsonify(ok=True)


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    print(f"AIR-BENCH pipeline UI on http://127.0.0.1:{port}")
    app.run(host="127.0.0.1", port=port, threaded=True)
