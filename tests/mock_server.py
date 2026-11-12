"""Servidor local que imita a Images API e a Batch API (para testar sem gastar créditos).

Imagens: POST /v1/images/generations. Comportamentos especiais, escolhidos por um marcador no prompt:
  FAIL429  -> responde 429 com Retry-After: 0 na 1ª tentativa e sucesso nas seguintes
  QUOTA    -> 429 insufficient_quota (não deve ser repetido)
  FAIL400  -> 400 com mensagem de erro no formato da OpenAI
  SLOW     -> demora 1,5 s para responder (para testar timeout)
  WHITE    -> imagem branca (fundo fora do chroma key)

Batch: POST /v1/files (multipart), POST /v1/batches, GET /v1/batches/<id>, GET /v1/files/<id>/content.
  ``batch_status`` é o status devolvido por GET /v1/batches/<id>. Quando vale "completed" (ou "expired"), o servidor
  processa o arquivo enviado: cada linha vira uma linha do arquivo de saída, ou do de erros se o prompt tiver FAIL400.
  ``drop_last_result`` omite o resultado da última linha (pedido que não terminou) e ``inject_output_lines``
  acrescenta linhas arbitrárias ao arquivo de saída (para testar um arquivo adulterado).
"""

from __future__ import annotations

import base64
import json
import re
import struct
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from openai_images.pricing import canonical, get_model, output_tokens, parse_size


def make_png(width: int, height: int, color=(0, 177, 64)) -> bytes:
    """PNG de cor sólida com um retângulo central diferente (só biblioteca padrão)."""
    row_bg = b"\x00" + bytes(color) * width
    x0, x1 = width // 3, 2 * width // 3
    row_fg = b"\x00" + bytes(color) * x0 + bytes((90, 90, 110)) * (x1 - x0) + bytes(color) * (width - x1)
    raw = b"".join(row_fg if height // 3 <= y < 2 * height // 3 else row_bg for y in range(height))

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b""))


def parse_multipart(content_type: str, body: bytes) -> dict[str, tuple[str | None, bytes]]:
    """``campo -> (nome do arquivo ou None, conteúdo)`` de um corpo multipart/form-data simples."""
    boundary = re.search(r"boundary=(\S+)", content_type).group(1).encode()
    fields = {}
    for part in body.split(b"--" + boundary):
        if b"\r\n\r\n" not in part:
            continue
        head, _, content = part.partition(b"\r\n\r\n")
        content = content.removesuffix(b"\r\n")
        name = re.search(rb'name="([^"]*)"', head).group(1).decode()
        filename = re.search(rb'filename="([^"]*)"', head)
        fields[name] = (filename.group(1).decode() if filename else None, content)
    return fields


class MockImagesServer:
    def __init__(self, api_key: str = "test-key"):
        self.api_key = api_key
        self.requests: list[dict] = []
        self._seen: set[str] = set()
        self.files: dict[str, bytes] = {}       # id -> conteúdo (envios e arquivos de saída)
        self.batches: dict[str, dict] = {}
        self.batch_status = "validating"
        self.drop_last_result = False
        self.inject_output_lines: list[dict] = []
        outer = self

        def image_payload(body: dict) -> dict:
            prompt = body["prompt"]
            width, height = parse_size(body["size"])
            color = (255, 255, 255) if "WHITE" in prompt else (0, 177, 64)
            png = base64.b64encode(make_png(width, height, color)).decode()
            n = body.get("n", 1)
            tokens = output_tokens(get_model(body["model"]), body["quality"], width, height) or 1000
            text_tokens = max(1, len(prompt) // 4)
            usage = {"input_tokens": text_tokens, "input_tokens_details": {"text_tokens": text_tokens, "image_tokens": 0},
                     "output_tokens": tokens * n, "total_tokens": text_tokens + tokens * n}
            return {"created": 1, "data": [{"b64_json": png} for _ in range(n)], "usage": usage,
                    "size": body["size"], "quality": body["quality"], "output_format": body.get("output_format", "png")}

        def finish_batch(batch: dict) -> None:
            """Processa o arquivo de pedidos e cria os arquivos de saída e de erros."""
            lines = [json.loads(l) for l in outer.files[batch["input_file_id"]].decode().splitlines() if l.strip()]
            if outer.drop_last_result:
                lines = lines[:-1]
            ok, failed = [], []
            for i, item in enumerate(lines, start=1):
                if "FAIL400" in item["body"]["prompt"]:
                    failed.append({"id": f"batch_req_{i}", "custom_id": item["custom_id"], "response": None,
                                   "error": {"code": "invalid_value", "message": "Invalid value: 'quality'"}})
                else:
                    ok.append({"id": f"batch_req_{i}", "custom_id": item["custom_id"], "error": None,
                               "response": {"status_code": 200, "request_id": f"req_batch_{i}",
                                            "body": image_payload(item["body"])}})
            ok.extend(outer.inject_output_lines)
            batch["output_file_id"] = f"file-out-{batch['id']}" if ok else None
            batch["error_file_id"] = f"file-err-{batch['id']}" if failed else None
            for file_id, rows in ((batch["output_file_id"], ok), (batch["error_file_id"], failed)):
                if file_id:
                    outer.files[file_id] = ("\n".join(json.dumps(r) for r in rows) + "\n").encode()
            batch["request_counts"] = {"total": len(lines), "completed": len(ok), "failed": len(failed)}
            batch["_done"] = True

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):   # silencioso
                pass

            def _reply(self, status, body: bytes, content_type="application/json", headers=None):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("x-request-id", f"req_{len(outer.requests)}")
                self.send_header("openai-processing-ms", "1234")
                for k, v in (headers or {}).items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                try:
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def _send(self, status, payload, headers=None):
                self._reply(status, json.dumps(payload).encode(), headers=headers)

            def _unauthorized(self) -> bool:
                if self.headers.get("Authorization") == f"Bearer {outer.api_key}":
                    return False
                self._send(401, {"error": {"message": "Incorrect API key provided", "code": "invalid_api_key"}})
                return True

            def do_GET(self):
                outer.requests.append({"method": "GET", "path": self.path, "auth": self.headers.get("Authorization"),
                                       "body": None})
                if self._unauthorized():
                    return
                if match := re.fullmatch(r"/v1/batches/([\w-]+)", self.path):
                    batch = outer.batches.get(match[1])
                    if batch is None:
                        return self._send(404, {"error": {"message": "No batch found", "code": "not_found"}})
                    batch["status"] = outer.batch_status
                    if batch["status"] in ("completed", "expired") and not batch.get("_done"):
                        finish_batch(batch)
                    return self._send(200, {k: v for k, v in batch.items() if not k.startswith("_")})
                if match := re.fullmatch(r"/v1/files/([\w-]+)/content", self.path):
                    if match[1] not in outer.files:
                        return self._send(404, {"error": {"message": "No such file", "code": "not_found"}})
                    return self._reply(200, outer.files[match[1]], content_type="application/octet-stream")
                self._send(404, {"error": {"message": "not found", "code": "not_found"}})

            def do_POST(self):
                raw = self.rfile.read(int(self.headers["Content-Length"]))
                record = {"method": "POST", "path": self.path, "auth": self.headers.get("Authorization"), "body": None}
                outer.requests.append(record)
                if self.path == "/v1/files":
                    if self._unauthorized():
                        return
                    fields = parse_multipart(self.headers["Content-Type"], raw)
                    filename, content = fields["file"]
                    file_id = f"file-in-{len(outer.files) + 1}"
                    outer.files[file_id] = content
                    record["body"] = {"purpose": fields["purpose"][1].decode(), "filename": filename, "file_id": file_id}
                    return self._send(200, {"id": file_id, "object": "file", "bytes": len(content), "filename": filename})
                body = json.loads(raw)
                record["body"] = body
                if self.path == "/v1/batches":
                    if self._unauthorized():
                        return
                    if body.get("input_file_id") not in outer.files:
                        return self._send(400, {"error": {"message": "Invalid input_file_id", "code": "invalid_value"}})
                    batch_id = f"batch_{len(outer.batches) + 1}"
                    outer.batches[batch_id] = {
                        "id": batch_id, "object": "batch", "endpoint": body["endpoint"],
                        "input_file_id": body["input_file_id"], "completion_window": body["completion_window"],
                        "status": "validating", "output_file_id": None, "error_file_id": None, "created_at": 1,
                        "expires_at": 2, "request_counts": {"total": 0, "completed": 0, "failed": 0}}
                    return self._send(200, {k: v for k, v in outer.batches[batch_id].items() if not k.startswith("_")})
                if self.path != "/v1/images/generations":
                    return self._send(404, {"error": {"message": "not found", "code": "not_found"}})
                if self._unauthorized():
                    return
                prompt = body["prompt"]
                if "QUOTA" in prompt:
                    return self._send(429, {"error": {"message": "You exceeded your current quota", "code": "insufficient_quota"}})
                if "FAIL400" in prompt:
                    return self._send(400, {"error": {"message": "Invalid value: 'quality'", "code": "invalid_value"}})
                if "FAIL429" in prompt and "FAIL429" not in outer._seen:
                    outer._seen.add("FAIL429")
                    return self._send(429, {"error": {"message": "Rate limit reached", "code": "rate_limit_exceeded"}},
                                      {"Retry-After": "0"})
                if "SLOW" in prompt:
                    time.sleep(1.5)
                self._send(200, image_payload(body))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.base_url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1"
        self._thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
