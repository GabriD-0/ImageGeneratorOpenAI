"""Batch API: metade do preço, resultado em até 24 h. Três etapas: ``submit``, ``status`` e ``fetch``.

Tudo o que as etapas seguintes precisam (id do batch, modelo, qualidade, ids dos pedidos) fica em
``batches/<data>/batch.json``; ``status`` e ``fetch`` recebem essa pasta.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from . import checks, report
from .client import ApiError, ImagesClient, image_payload
from .pricing import BATCH_DISCOUNT, batch_cost_from_usage, estimate

ENDPOINT = "/v1/images/generations"
COMPLETION_WINDOW = "24h"       # a API só aceita esta janela
MAX_REQUESTS = 50_000           # limite de pedidos por batch
STATE_FILE = "batch.json"
REQUESTS_FILE = "requests.jsonl"
RESULTS_FILE = "fetch.json"
PENDING = ("validating", "in_progress", "finalizing", "cancelling")   # ainda sem resultado final


# ----------------------------------------------------------------------------- submit


def build_requests(*, model: str, quality: str, prompt_text: str, size: str, count: int, output_format: str,
                   background: str | None) -> list[dict]:
    """Uma linha do .jsonl por imagem, todas com o mesmo corpo da chamada normal (n=1) e ``custom_id`` único."""
    slug = re.sub(r"[^A-Za-z0-9.]+", "_", f"{model}_{quality}")
    body = image_payload(model=model, prompt=prompt_text, size=size, quality=quality, n=1,
                         output_format=output_format, background=background)
    return [{"custom_id": f"{slug}_{k:03d}", "method": "POST", "url": ENDPOINT, "body": body}
            for k in range(1, count + 1)]


def to_jsonl(requests: list[dict]) -> bytes:
    return ("\n".join(json.dumps(r, ensure_ascii=False) for r in requests) + "\n").encode("utf-8")


def run_submit(*, client: ImagesClient | None, model: str, quality: str, size: str, count: int, prompt_text: str,
               prompt_tokens: int, output_format: str, background: str | None, out_dir: Path, max_cost: float,
               assume_yes: bool, dry_run: bool) -> int:
    est = estimate(model, quality, size, prompt_tokens)   # valida modelo, qualidade e tamanho
    if est.batch_usd is None:
        raise ValueError(f"{est.model} não tem preço de Batch API; use `generate` ou `compare`")
    batch_total, normal_total = est.batch_usd * count, est.total_usd * count

    requests = build_requests(model=model, quality=quality, prompt_text=prompt_text, size=size, count=count,
                              output_format=output_format, background=background)
    print(f"{model} · quality={quality} · {size} · {count} imagem(ns) em 1 batch")
    print(f"Custo estimado no Batch: {report.usd(batch_total)} "
          f"(no modo normal: {report.usd(normal_total)}; {est.output_tokens or 'tabela'} tokens de saída por imagem)")
    if dry_run:
        print(f"\nPrimeira linha do {REQUESTS_FILE}:\n{json.dumps(requests[0], ensure_ascii=False)}")
        print("\n(--dry-run) nada foi enviado.")
        return 0
    if batch_total > max_cost:
        print(f"erro: estimativa {report.usd(batch_total)} acima de --max-cost {report.usd(max_cost)}.",
              file=sys.stderr)
        return 2
    if not assume_yes:
        if not sys.stdin.isatty():
            print("Execute com --yes para confirmar os gastos sem perguntar.", file=sys.stderr)
            return 1
        if input("Enviar o batch (as imagens serão cobradas quando ficarem prontas)? [s/N] "
                 ).strip().lower() not in {"s", "sim", "y", "yes"}:
            return 1
    assert client is not None

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir, suffix = out_dir / stamp, 2
    while run_dir.exists():   # dois envios no mesmo segundo não podem sobrescrever o batch.json um do outro
        run_dir, suffix = out_dir / f"{stamp}-{suffix}", suffix + 1
    run_dir.mkdir(parents=True)
    content = to_jsonl(requests)
    (run_dir / REQUESTS_FILE).write_bytes(content)

    print("Enviando o arquivo de pedidos ...")
    file_id = client.upload_file(content, REQUESTS_FILE, purpose="batch")
    try:
        batch = client.create_batch(file_id, ENDPOINT, COMPLETION_WINDOW)
    except ApiError:
        print(f"O arquivo foi enviado ({file_id}), mas o batch não foi criado.", file=sys.stderr)
        raise
    state = {
        "batch_id": batch["id"], "input_file_id": file_id, "created": datetime.now().isoformat(timespec="seconds"),
        "model": model, "quality": quality, "size": size, "count": count, "output_format": output_format,
        "background": background, "custom_ids": [r["custom_id"] for r in requests], "prompt": prompt_text,
        "estimate_usd_batch": batch_total, "estimate_usd_normal": normal_total,
    }
    (run_dir / STATE_FILE).write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Batch criado: {batch['id']} (status: {batch.get('status')})\nPasta: {run_dir}\n"
          f"  acompanhar: python -m openai_images batch status {run_dir}\n"
          f"  baixar:     python -m openai_images batch fetch {run_dir}")
    return 0


# ----------------------------------------------------------------------------- status


def load_state(batch_dir: Path) -> dict:
    path = batch_dir / STATE_FILE
    if not path.is_file():
        raise FileNotFoundError(f"{path} não existe; passe a pasta criada por `batch submit`")
    return json.loads(path.read_text(encoding="utf-8"))


def _when(timestamp: int | None) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%d/%m/%Y %H:%M") if timestamp else "—"


def _counts(batch: dict) -> str:
    c = batch.get("request_counts") or {}
    return f"{c.get('completed', 0)} concluído(s), {c.get('failed', 0)} com falha, {c.get('total', '?')} no total"


def _batch_errors(batch: dict) -> list[str]:
    data = (batch.get("errors") or {}).get("data") or []
    return [str(e.get("message") or e) if isinstance(e, dict) else str(e) for e in data]


def run_status(*, client: ImagesClient, batch_dir: Path) -> int:
    state = load_state(batch_dir)
    batch = client.get_batch(state["batch_id"])
    status = batch.get("status")
    print(f"Batch {batch.get('id')}: {status}\n  pedidos:  {_counts(batch)}\n"
          f"  criado:   {_when(batch.get('created_at'))}\n  expira:   {_when(batch.get('expires_at'))}")
    for message in _batch_errors(batch):
        print(f"  erro: {message}")
    if status in PENDING:
        print("\nAinda em andamento (a janela é de até 24 h). Rode `batch status` de novo mais tarde.")
    elif status == "failed":
        print("\nO batch falhou na validação: nada foi gerado nem cobrado.")
        return 1
    else:
        if status == "expired":
            print("\nA janela de 24 h acabou: só os pedidos concluídos foram cobrados e estão no resultado.")
        print(f"\nBaixe as imagens com: python -m openai_images batch fetch {batch_dir}")
    return 0


# ----------------------------------------------------------------------------- fetch


def interpret(item: dict) -> dict:
    """Uma linha do arquivo de saída ou de erros -> imagem, uso e erro (nos dois formatos que a API usa)."""
    response = item.get("response") if isinstance(item.get("response"), dict) else {}
    body = response.get("body") if isinstance(response.get("body"), dict) else {}
    error = item.get("error") or body.get("error")
    status_code = response.get("status_code")
    if error is None and status_code not in (None, 200):
        error = {"message": f"HTTP {status_code}"}
    image = None
    if error is None:
        data = body.get("data")
        first = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else {}
        if not first.get("b64_json"):
            error = {"message": "resposta sem imagem (b64_json)"}
        else:
            try:
                image = base64.b64decode(first["b64_json"], validate=True)
            except (binascii.Error, ValueError):
                error = {"message": "imagem em base64 inválido"}
    message = None
    if error is not None:
        text = (error.get("message") if isinstance(error, dict) else None) or str(error)
        code = error.get("code") if isinstance(error, dict) else None
        message = f"{code}: {text}" if code else text
    return {"image": image, "usage": body.get("usage"), "request_id": response.get("request_id"), "error": message}


def run_fetch(*, client: ImagesClient, batch_dir: Path, min_height: int,
              key_color: tuple[int, int, int] | None) -> int:
    state = load_state(batch_dir)
    batch = client.get_batch(state["batch_id"])
    status = batch.get("status")
    if status in PENDING:
        print(f"O batch ainda está '{status}' ({_counts(batch)}). Tente de novo mais tarde.", file=sys.stderr)
        return 1
    if not (batch.get("output_file_id") or batch.get("error_file_id")):
        reasons = "; ".join(_batch_errors(batch))
        print(f"O batch terminou como '{status}' sem arquivo de resultado." + (f" {reasons}" if reasons else ""),
              file=sys.stderr)
        return 1

    expected = state["custom_ids"]
    known = {c for c in expected if isinstance(c, str)}
    results: dict[str, dict] = {}   # custom_id -> interpret(); vale a primeira ocorrência (saída antes de erros)
    for key in ("output_file_id", "error_file_id"):
        if not batch.get(key):
            continue
        text = client.download_file(batch[key]).decode("utf-8", "replace")
        for number, line in enumerate(filter(str.strip, text.splitlines()), start=1):
            try:
                item = json.loads(line)
            except ValueError:
                print(f"aviso: {key}, linha {number} ignorada (JSON inválido)", file=sys.stderr)
                continue
            custom_id = item.get("custom_id") if isinstance(item, dict) else None
            if not isinstance(custom_id, str) or custom_id not in known:
                print(f"aviso: {key}, linha {number} ignorada (custom_id desconhecido: {custom_id!r})", file=sys.stderr)
                continue
            results.setdefault(custom_id, interpret(item))

    images, failures, rows = [], [], []
    for custom_id in expected:
        result = results.get(custom_id)
        if result is None:
            failures.append({"custom_id": custom_id, "error": f"sem resultado (batch {status})"})
            continue
        if result["error"] is not None:
            failures.append({"custom_id": custom_id, "error": result["error"]})
            continue
        name = f"{custom_id}.{state['output_format']}"
        (batch_dir / name).write_bytes(result["image"])
        info = checks.analyze(result["image"], min_height=min_height, key_color=key_color)
        cost = batch_cost_from_usage(state["model"], result["usage"])
        images.append({"custom_id": custom_id, "file": name, "cost_usd": cost, "usage": result["usage"],
                       "request_id": result["request_id"], **info})
        notes = []
        if info["meets_height"] is False:
            notes.append(f"⚠ altura {info['height']} < {min_height}")
        if info["key_ok"] is False:
            notes.append(f"⚠ só {info['key_share']:.0%} da borda é chroma key")
        elif info["key_share"] is not None:
            notes.append(f"fundo verde {info['key_share']:.0%}")
        rows.append([name, f"{info['width']}x{info['height']}" if info["width"] else "—",
                     report.usd(cost) if cost is not None else "—", "; ".join(notes) or "—"])

    known_costs = [i["cost_usd"] for i in images if i["cost_usd"] is not None]
    total = sum(known_costs)
    partial = len(known_costs) < len(images)   # alguma resposta veio sem usage
    (batch_dir / RESULTS_FILE).write_text(json.dumps({
        "batch_id": state["batch_id"], "status": status, "fetched": datetime.now().isoformat(timespec="seconds"),
        "model": state["model"], "quality": state["quality"], "size": state["size"], "requested": len(expected),
        "succeeded": len(images), "cost_usd": total, "cost_is_partial": partial,
        "estimate_usd_batch": state.get("estimate_usd_batch"), "images": images, "failures": failures,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    if rows:
        print(report.table(["imagem", "tamanho", "custo (batch)", "checagem"], rows))
    print(f"\n{len(images)} de {len(expected)} imagem(ns) salvas em {batch_dir} (batch {status}).")
    if known_costs:
        print(f"Custo real no Batch: {report.usd(total)}{' (parcial: faltou usage em alguma resposta)' if partial else ''}"
              f" · no modo normal seria ≈ {report.usd(total / BATCH_DISCOUNT)}"
              f" · estimado no envio: {report.usd(state.get('estimate_usd_batch'))}")
    elif images:
        print(f"Custo real indisponível (as respostas vieram sem usage); estimado no envio: "
              f"{report.usd(state.get('estimate_usd_batch'))}")
    if failures:
        print(f"\n{len(failures)} sem imagem:")
        for failure in failures:
            print(f"  {failure['custom_id']}: {failure['error']}")
    return 0 if not failures else 1
