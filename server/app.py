"""Yurameki simulation server — line-delimited JSON over TCP.

One JSON object per line: {"id","method","params"} -> {"id","result"|"error"}.
Methods:
  ping                       -> {"pong": True, ...}
  simulate {testdata,out,start,end,overrides} -> engine result dict
  shutdown                   -> {"bye": True} then the server exits

Mirrors tanabata's socket pattern, but the simulator is Blender-independent.
Run:   python app.py [--host 127.0.0.1] [--port 7780]
"""
from __future__ import annotations
import argparse, json, os, socket, sys, traceback

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
import engine

HOST = "127.0.0.1"
PORT = 7780


def _handle(req: dict) -> dict:
    method = req.get("method")
    params = req.get("params") or {}
    if method == "ping":
        return {"pong": True, "engine": "yurameki", "cwd": _HERE}
    if method == "simulate":
        testdata = params.get("testdata", os.path.join(os.path.dirname(_HERE), "testdata"))
        out = params.get("out", os.path.join(testdata, "hair_sim.npz"))
        return engine.run_from_testdata(
            testdata, out,
            start=params.get("start"), end=params.get("end"),
            overrides=params.get("overrides"),
        )
    if method == "shutdown":
        return {"bye": True}
    raise ValueError(f"unknown method: {method!r}")


def serve(host=HOST, port=PORT):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(4)
    print(f"[yurameki/server] listening on {host}:{port}", flush=True)
    running = True
    while running:
        conn, _addr = srv.accept()
        with conn:
            buf = b""
            while not buf.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                buf += chunk
            if not buf:
                continue
            try:
                req = json.loads(buf.decode("utf-8"))
                result = _handle(req)
                resp = {"id": req.get("id"), "result": result}
                if req.get("method") == "shutdown":
                    running = False
            except Exception as exc:  # noqa: BLE001
                resp = {"id": None, "error": str(exc),
                        "trace": traceback.format_exc()}
            conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
    srv.close()
    print("[yurameki/server] stopped", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    serve(args.host, args.port)
