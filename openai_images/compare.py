from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from . import checks, report
from .client import ApiError, ImagesClient
from .pricing import MODELS, cost_from_usage, estimate

# Candidatas ao requisito do projeto (retrato, altura >= 1600). O 2.6 fica de fora por padrão: não consta
# na referência da Images API e custa mais em low/medium; inclua com --configs se quiser testá-lo.
DEFAULT_CONFIGS = ("gpt-image-2.5-flare:low,gpt-image-2.5-flare:medium,gpt-image-2.5-flare:high,"
                   "gpt-image-2.5-sunburst:medium,gpt-image-2.5-sunburst:high,gpt-image-2.5-sunburst:xhigh,"
                   "gpt-image-2:medium")


@dataclass(frozen=True)
class Config:
    model: str
    quality: str
    complexity: str | None = None   # só o 2.6

    @property
    def label(self) -> str:
        return ":".join(p for p in (self.model, self.quality, self.complexity) if p)

    @property
    def slug(self) -> str:
        return re.sub(r"[^A-Za-z0-9.]+", "_", self.label)


def parse_configs(text: str) -> list[Config]:
    """``modelo:qualidade[:complexidade]`` separados por vírgula."""
    configs = []
    for item in filter(None, (t.strip() for t in text.split(","))):
        parts = item.split(":")
        if len(parts) not in (2, 3):
            raise ValueError(f"configuração inválida: {item!r} (use modelo:qualidade, ex.: gpt-image-2.5-flare:high)")
        if parts[0] not in MODELS:
            raise ValueError(f"modelo desconhecido: {parts[0]!r}. Conhecidos: {', '.join(MODELS)}")
        if len(parts) == 3 and not (parts[0].startswith("gpt-image-2.6") and parts[2] in ("low", "medium", "high")):
            raise ValueError(f"{item!r}: a complexidade (low, medium ou high) só existe nos modelos gpt-image-2.6-*")
        configs.append(Config(*parts))
    if not configs:
        raise ValueError("nenhuma configuração informada")
    if len({c.label for c in configs}) != len(configs):
        raise ValueError("há configurações repetidas")
    return configs


def run_compare(*, configs: list[Config], prompt_text: str, prompt_tokens: int, size: str, samples: int,
                out_dir: Path, budget: float, assume_yes: bool, dry_run: bool, min_height: int,
                key_color: tuple[int, int, int] | None, output_format: str, background: str | None,
                client: ImagesClient | None) -> int:
    plan = []
    for cfg in configs:
        est = estimate(cfg.model, cfg.quality, size, prompt_tokens)   # valida modelo, qualidade e tamanho
        plan.append((cfg, est))
    total = sum(est.total_usd for _, est in plan) * samples

    print(f"Plano: {len(plan)} configurações x {samples} amostra(s), {size}\n")
    print(report.table(["config", "tokens/img", "custo estimado/img", "batch (50%)"], [
        [cfg.label, str(est.output_tokens or "—"), report.usd(est.total_usd),
         report.usd(est.batch_usd) if est.batch_usd is not None else "indisponível"]
        for cfg, est in plan]))
    print(f"\nCusto total estimado: {report.usd(total)} (orçamento máximo: {report.usd(budget)})")
    if dry_run:
        print("(--dry-run) nenhuma chamada foi feita.")
        return 0
    if total > budget:
        print("erro: a estimativa passa do orçamento. Reduza --samples/--configs ou aumente --budget.",
              file=sys.stderr)
        return 2
    if not assume_yes:
        if not sys.stdin.isatty():
            print("Execute com --yes para confirmar os gastos sem perguntar.", file=sys.stderr)
            return 1
        if input("Fazer as chamadas (isto gera cobrança)? [s/N] ").strip().lower() not in {"s", "sim", "y", "yes"}:
            return 1
    assert client is not None

    run_dir = out_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    records, spent = [], 0.0
    for index, (cfg, est) in enumerate(plan, start=1):
        label = f"[{index}/{len(plan)}] {cfg.label}"
        needed = est.total_usd * samples
        if spent + needed > budget:
            print(f"{label}: pulada (gasto {report.usd(spent)} + {report.usd(needed)} passaria do orçamento)")
            records.append(_error_record(cfg, size, samples, "pulada: orçamento"))
            continue
        print(f"{label} ...", end=" ", flush=True)
        extra = {"background": background, "complexity": cfg.complexity}
        try:
            resp = client.generate(model=cfg.model, prompt=prompt_text, size=size, quality=cfg.quality,
                                   n=samples, output_format=output_format, **extra)
        except ApiError as err:
            print(f"ERRO ({err.status or err.kind}): {err}")
            records.append(_error_record(cfg, size, samples, f"{err.status or err.kind}: {err}"))
            if err.status in (401, 403) or err.code == "insufficient_quota":
                print("Credencial ou saldo recusados: interrompendo a comparação.", file=sys.stderr)
                break
            continue

        images = []
        for k, data in enumerate(resp.images, start=1):
            name = f"{cfg.slug}_{k}.{output_format}"
            (run_dir / name).write_bytes(data)
            images.append({"file": name, **checks.analyze(data, min_height=min_height, key_color=key_color)})
        cost = cost_from_usage(cfg.model, resp.usage)
        spent += cost if cost is not None else est.total_usd * len(resp.images)
        per_image = cost / len(resp.images) if cost is not None else None
        tokens = (resp.usage or {}).get("output_tokens")
        records.append({
            "label": cfg.label, "model": cfg.model, "quality": cfg.quality, "complexity": cfg.complexity,
            "size": size, "samples": samples, "estimate_usd": est.total_usd,
            "cost_usd": cost, "cost_per_image": per_image,
            "tokens_per_image": round(tokens / len(resp.images)) if tokens else None,
            "latency_s": round(resp.latency_s, 1), "processing_ms": resp.processing_ms,
            "request_id": resp.request_id, "usage": resp.usage, "images": images, "error": None,
        })
        shown = report.usd(per_image) if per_image is not None else f"{report.usd(est.total_usd)} (est.)"
        print(f"{resp.latency_s:.0f}s · {shown}/img (estimado {report.usd(est.total_usd)})")

    data = {"created": datetime.now().isoformat(timespec="seconds"), "prompt": prompt_text, "size": size,
            "samples": samples, "min_height": min_height, "records": records}
    (run_dir / report.RESULTS_FILE).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    report.write_sheet(run_dir, data)
    report.write_ratings_template(run_dir, records)

    print("\n" + report.table(report.SUMMARY_HEADERS, report.summary_rows(records)))
    print(f"\nGasto real nesta comparação: {report.usd(spent)}")
    print(f"Pasta: {run_dir}\n  1. abra {report.SHEET_FILE} e olhe as imagens\n"
          f"  2. dê notas de 1 a 5 em {report.RATINGS_FILE}\n"
          f"  3. python -m openai_images rank {run_dir}")
    return 0 if any(not r.get("error") for r in records) else 1


def _error_record(cfg: Config, size: str, samples: int, message: str) -> dict:
    return {"label": cfg.label, "model": cfg.model, "quality": cfg.quality, "complexity": cfg.complexity,
            "size": size, "samples": samples, "images": [], "error": message}
