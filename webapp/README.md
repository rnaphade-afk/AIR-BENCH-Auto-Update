# Pipeline Web UI

A thin browser front-end for orchestrating the AIR-BENCH **setup** and **update**
pipelines, with an integrated JSON editor for the human-in-the-loop review
checkpoints. It is a driver around the existing CLIs.

## Run

```bash
venv/bin/pip install -r requirements.txt   # installs flask
venv/bin/python webapp/server.py           # http://127.0.0.1:5000
```

Set `PORT` to change the port.

## Use

1. Pick a tool (`pipeline.py` or `setup.py`) and adjust the flags (toggles fill in
   common ones; the text box accepts any CLI flag).
2. **Start run.** Logs stream live in the bottom panel.
3. When the run hits a review checkpoint it pauses, opens the JSON file in the
   editor, and shows a banner. Edit the file (it is validated as JSON), then
   **Save & Continue** to resume.
4. Repeat until the run reports `done`. Other checkpoint files for the run are
   listed under "Run-dir checkpoints" and can be opened any time.

## Notes

- `--yes` is never injected; the pause is how the UI gets to edit. (You may still
  pass `--yes` yourself to blast through without pausing.)
- One run at a time. File reads/writes are sandboxed to the repository root.
- Local, single-user tool — no authentication.
- The editor (CodeMirror) loads from a CDN; for fully offline use, swap it for a
  plain textarea in `index.html`.
