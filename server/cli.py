"""Tiny client for the yurameki simulation server.

Examples:
  python cli.py ping
  python cli.py simulate --start 0 --end 90 --out ../testdata/hair_cli.npz
  python cli.py shutdown
"""
from __future__ import annotations
import argparse, json, socket, sys

HOST, PORT = "127.0.0.1", 7780


def call(method, params=None, host=HOST, port=PORT, timeout=600):
    req = {"id": 1, "method": method, "params": params or {}}
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall((json.dumps(req) + "\n").encode("utf-8"))
        buf = b""
        while not buf.endswith(b"\n"):
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
    return json.loads(buf.decode("utf-8"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("method", choices=["ping", "simulate", "shutdown"])
    ap.add_argument("--start", type=int)
    ap.add_argument("--end", type=int)
    ap.add_argument("--out")
    ap.add_argument("--host", default=HOST)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    params = {}
    if args.method == "simulate":
        if args.start is not None: params["start"] = args.start
        if args.end is not None: params["end"] = args.end
        if args.out: params["out"] = args.out
    resp = call(args.method, params, args.host, args.port)
    print(json.dumps(resp, indent=2))
    sys.exit(0 if "result" in resp else 1)
