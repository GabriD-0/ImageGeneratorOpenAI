from __future__ import annotations

import base64
import json
import os
import random
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from . import ROOT

DEFAULT_BASE_URL = "https://api.openai.com/v1"
_RETRY_STATUS = {429, 500, 502, 503, 504}
_NO_RETRY_CODES = {"insufficient_quota"}   # 429 por falta de saldo: repetir não adianta
_USER_AGENT = "ImageGeneratorOpenAI/1.0 (urllib)"

_TIMEOUT_GENERATE = "a imagem pode ter sido gerada e cobrada mesmo assim (confira o painel de uso)."
_TIMEOUT_CREATE_BATCH = ("o batch pode ter sido criado mesmo assim (confira o painel de batches antes de "
                         "enviar de novo).")
_TIMEOUT_READ = "nada foi alterado; pode tentar de novo."


class ApiError(Exception):
    """Erro da API ou de rede, com os campos úteis para diagnosticar."""

    def __init__(self, message: str, *, status: int | None = None, code: str | None = None,
                 kind: str = "api", request_id: str | None = None):
        super().__init__(message)
        self.status, self.code, self.kind, self.request_id = status, code, kind, request_id


@dataclass
class ImageResponse:
    images: list[bytes]
    usage: dict | None
    latency_s: float
    request_id: str | None
    processing_ms: int | None
    meta: dict   # created, size, quality, background, output_format (como a API devolveu)


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """Carrega ``CHAVE=valor`` de um .env simples para o ambiente, sem sobrescrever o que já existe."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip().removeprefix("export ").strip(), value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def get_api_key() -> str:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise ApiError("OPENAI_API_KEY não definida. Exporte a variável ou crie o arquivo .env "
                       "(veja .env.example).", kind="config")
    return key


def image_payload(*, model: str, prompt: str, size: str, quality: str, n: int = 1,
                  output_format: str = "png", **extra) -> dict:
    """Corpo de um pedido de geração. ``extra`` (background, moderation, complexity...) entra se não for None.

    É o mesmo corpo da chamada normal e de cada linha do arquivo do Batch.
    """
    payload = {"model": model, "prompt": prompt, "size": size, "quality": quality, "n": n,
               "output_format": output_format}
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def _error_from_http(err: urllib.error.HTTPError) -> ApiError:
    request_id = err.headers.get("x-request-id")
    code = message = None
    try:
        body = json.loads(err.read().decode("utf-8", "replace"))
        detail = body.get("error") or {}
        message, code = detail.get("message"), detail.get("code") or detail.get("type")
    except (ValueError, AttributeError):
        pass
    return ApiError(message or f"HTTP {err.code}: {err.reason}", status=err.code, code=code,
                    request_id=request_id)


class ImagesClient:
    def __init__(self, api_key: str, *, base_url: str | None = None, timeout: float = 300,
                 max_retries: int = 3):
        self._api_key = api_key
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries

    def __repr__(self) -> str:   # nunca expõe a chave
        return f"ImagesClient(base_url={self.base_url!r})"

    # ------------------------------------------------------------------------- geração (chamada normal)

    def generate(self, *, model: str, prompt: str, size: str, quality: str, n: int = 1,
                 output_format: str = "png", **extra) -> ImageResponse:
        """Gera ``n`` imagens. ``extra`` (background, moderation, complexity...) entra no corpo se não for None."""
        payload = image_payload(model=model, prompt=prompt, size=size, quality=quality, n=n,
                                output_format=output_format, **extra)
        raw, headers, latency = self._request(
            "POST", "/images/generations", body=json.dumps(payload).encode("utf-8"),
            content_type="application/json", timeout_note=_TIMEOUT_GENERATE)
        data = json.loads(raw.decode("utf-8"))
        items = data.get("data") or []
        images = [base64.b64decode(item["b64_json"]) for item in items if item.get("b64_json")]
        if not images:
            raise ApiError("a resposta não trouxe nenhuma imagem (b64_json)", kind="empty",
                           request_id=headers.get("x-request-id"))
        processing = headers.get("openai-processing-ms")
        return ImageResponse(
            images=images,
            usage=data.get("usage"),
            latency_s=latency,
            request_id=headers.get("x-request-id"),
            processing_ms=int(processing) if processing and processing.isdigit() else None,
            meta={k: data.get(k) for k in ("created", "size", "quality", "background", "output_format")},
        )

    # ------------------------------------------------------------------------- Batch API

    def upload_file(self, content: bytes, filename: str, *, purpose: str) -> str:
        """Envia um arquivo (Files API, multipart) e devolve o ``id`` dele."""
        boundary = f"----imagegen{os.urandom(12).hex()}"
        body = b"".join((
            f'--{boundary}\r\nContent-Disposition: form-data; name="purpose"\r\n\r\n{purpose}\r\n'.encode(),
            (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
             "Content-Type: application/octet-stream\r\n\r\n").encode(),
            content,
            f"\r\n--{boundary}--\r\n".encode(),
        ))
        raw, _, _ = self._request("POST", "/files", body=body,
                                  content_type=f"multipart/form-data; boundary={boundary}",
                                  timeout_note=_TIMEOUT_READ)
        return json.loads(raw.decode("utf-8"))["id"]

    def create_batch(self, input_file_id: str, endpoint: str, completion_window: str = "24h") -> dict:
        payload = {"input_file_id": input_file_id, "endpoint": endpoint, "completion_window": completion_window}
        raw, _, _ = self._request("POST", "/batches", body=json.dumps(payload).encode("utf-8"),
                                  content_type="application/json", timeout_note=_TIMEOUT_CREATE_BATCH)
        return json.loads(raw.decode("utf-8"))

    def get_batch(self, batch_id: str) -> dict:
        raw, _, _ = self._request("GET", f"/batches/{urllib.parse.quote(batch_id, safe='')}",
                                  timeout_note=_TIMEOUT_READ)
        return json.loads(raw.decode("utf-8"))

    def download_file(self, file_id: str) -> bytes:
        raw, _, _ = self._request("GET", f"/files/{urllib.parse.quote(file_id, safe='')}/content",
                                  timeout_note=_TIMEOUT_READ)
        return raw

    # ------------------------------------------------------------------------- HTTP

    def _request(self, method: str, path: str, *, body: bytes | None = None, content_type: str | None = None,
                 timeout_note: str) -> tuple[bytes, object, float]:
        """Faz a requisição com repetição segura; devolve (corpo, cabeçalhos, latência da última tentativa).

        Repete só 429 (exceto falta de saldo), 5xx e falhas antes de conectar. **Não repete timeout de leitura**:
        o pedido pode já ter sido processado (e cobrado); ``timeout_note`` diz o que isso significa na chamada.
        """
        headers = {"Authorization": f"Bearer {self._api_key}", "User-Agent": _USER_AGENT}
        if content_type:
            headers["Content-Type"] = content_type
        timeout_message = f"timeout após {self.timeout:g}s; {timeout_note} Não repeti automaticamente."

        attempt = 0
        while True:
            request = urllib.request.Request(f"{self.base_url}{path}", data=body, method=method, headers=headers)
            started = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                    raw = resp.read()
                    response_headers = resp.headers
                break
            except urllib.error.HTTPError as err:
                error = _error_from_http(err)
                retry_after = err.headers.get("Retry-After")
                if (err.code in _RETRY_STATUS and error.code not in _NO_RETRY_CODES
                        and attempt < self.max_retries):
                    self._sleep(attempt, retry_after)
                    attempt += 1
                    continue
                raise error from None
            except urllib.error.URLError as err:
                reason = err.reason
                if isinstance(reason, (socket.gaierror, ConnectionRefusedError)) and attempt < self.max_retries:
                    self._sleep(attempt, None)   # nem chegou a conectar: seguro repetir
                    attempt += 1
                    continue
                if isinstance(reason, (socket.timeout, TimeoutError)):
                    raise ApiError(timeout_message, kind="timeout") from None
                raise ApiError(f"falha de rede: {reason}", kind="network") from None
            except (socket.timeout, TimeoutError):   # estouro durante a leitura da resposta
                raise ApiError(timeout_message, kind="timeout") from None

        return raw, response_headers, time.monotonic() - started

    @staticmethod
    def _sleep(attempt: int, retry_after: str | None) -> None:
        try:
            delay = float(retry_after) if retry_after else 0.0
        except ValueError:
            delay = 0.0
        if not delay:
            delay = min(2 ** attempt * 2, 30) + random.uniform(0, 1)
        time.sleep(min(delay, 60))
