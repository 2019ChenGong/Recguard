#!/usr/bin/env python3
"""Minimal OpenAI-compatible /v1/embeddings server backed by sentence-transformers.

Used to give self-hosted memory frameworks that require an embedding endpoint
(Letta) a local model when the vLLM server only serves chat completions.
Endpoints: GET /v1/models, POST /v1/embeddings ({"input": str|[str], "model": ...}).

    PYTHONNOUSERSITE=1 python scripts/local_embedding_server.py --port 8285 \
        --model sentence-transformers/all-MiniLM-L6-v2
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8285)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(args.model)
    dim = model.get_sentence_embedding_dimension()
    lock = threading.Lock()
    model_id = args.model.split("/")[-1]
    print(f"embedding server: model={args.model} dim={dim} http://{args.host}:{args.port}/v1", flush=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):  # noqa: N802
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                self._send(200, {"object": "list", "data": [
                    {"id": model_id, "object": "model", "owned_by": "local", "embedding_dim": dim}
                ]})
            elif self.path.rstrip("/") in ("/health", "/v1/health", ""):
                self._send(200, {"status": "ok", "dim": dim})
            else:
                self._send(404, {"error": f"unknown path {self.path}"})

        def do_POST(self):  # noqa: N802
            if self.path.rstrip("/") not in ("/v1/embeddings", "/embeddings"):
                self._send(404, {"error": f"unknown path {self.path}"})
                return
            n = int(self.headers.get("Content-Length") or 0)
            try:
                req = json.loads(self.rfile.read(n) or b"{}")
                inp = req.get("input", "")
                texts = [inp] if isinstance(inp, str) else [str(t) for t in inp]
                with lock:
                    vecs = model.encode(texts, normalize_embeddings=True)
                data = [
                    {"object": "embedding", "index": i, "embedding": [float(x) for x in v]}
                    for i, v in enumerate(vecs)
                ]
                n_tok = sum(len(t.split()) for t in texts)
                self._send(200, {"object": "list", "data": data, "model": req.get("model") or model_id,
                                 "usage": {"prompt_tokens": n_tok, "total_tokens": n_tok}})
            except Exception as exc:  # noqa: BLE001
                self._send(500, {"error": str(exc)})

        def log_message(self, fmt, *a):  # quiet
            return

    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
