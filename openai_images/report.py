from __future__ import annotations

import csv
import html
import json
import statistics
from pathlib import Path

RATINGS_FILE = "ratings.csv"
RESULTS_FILE = "results.json"
SHEET_FILE = "sheet.html"
RANKING_FILE = "ranking.md"


def usd(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"US$ {value:.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> str:
    """Tabela de texto alinhada (colunas separadas por ' | ')."""
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)]
    def line(cells):
        return " | ".join(str(c).ljust(w) for c, w in zip(cells, widths)).rstrip()
    return "\n".join([line(headers), "-+-".join("-" * w for w in widths), *(line(r) for r in rows)])


def cost_per_image(record: dict) -> float | None:
    """Custo real por imagem; se a resposta não trouxe ``usage``, cai para a estimativa."""
    real = record.get("cost_per_image")
    return real if real is not None else record.get("estimate_usd")


def summary_rows(records: list[dict]) -> list[list[str]]:
    rows = []
    for r in records:
        if r.get("error"):
            rows.append([r["label"], "—", "—", "—", "—", f"ERRO: {r['error'][:60]}"])
            continue
        imgs = r["images"]
        shares = [i["key_share"] for i in imgs if i.get("key_share") is not None]
        rows.append([
            r["label"],
            str(r.get("tokens_per_image") or "—"),
            usd(cost_per_image(r)) + ("" if r.get("cost_per_image") is not None else " (est.)"),
            f"{r['latency_s']:.0f}s",
            f"{imgs[0]['width']}x{imgs[0]['height']}" if imgs and imgs[0].get("width") else "—",
            f"fundo verde {statistics.mean(shares):.0%}" if shares else "—",
        ])
    return rows


SUMMARY_HEADERS = ["config", "tokens/img", "custo/img", "tempo", "tamanho", "checagem"]


# ----------------------------------------------------------------------------- folha de contato e notas


def write_ratings_template(run_dir: Path, records: list[dict]) -> Path:
    path = run_dir / RATINGS_FILE
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["image", "config", "rating", "notes"])
        for r in records:
            for img in r.get("images", []):
                writer.writerow([img["file"], r["label"], "", ""])
    return path


def write_sheet(run_dir: Path, data: dict) -> Path:
    cards = []
    for r in data["records"]:
        if r.get("error"):
            cards.append(f'<figure class="card err"><figcaption><b>{html.escape(r["label"])}</b>'
                         f'<br>erro: {html.escape(r["error"])}</figcaption></figure>')
            continue
        for img in r["images"]:
            facts = [usd(cost_per_image(r)) + ("" if r.get("cost_per_image") is not None else " (est.)"),
                     f"{img.get('width')}x{img.get('height')}", f"{r['latency_s']:.0f}s"]
            if img.get("key_share") is not None:
                facts.append(f"borda verde {img['key_share']:.0%}")
            warn = ""
            if img.get("meets_height") is False:
                warn = '<span class="bad">altura abaixo do mínimo</span>'
            elif img.get("key_ok") is False:
                warn = '<span class="bad">fundo fora do chroma key</span>'
            name = html.escape(img["file"])
            cards.append(f'<figure class="card"><a href="{name}"><img src="{name}" loading="lazy" alt="{name}"></a>'
                         f'<figcaption><b>{html.escape(r["label"])}</b><br>{" · ".join(map(html.escape, facts))}'
                         f'<br><small>{name}</small> {warn}</figcaption></figure>')
    body = f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8"><title>Comparação de custo-benefício</title>
<style>
:root {{ color-scheme: light dark; --bg:#fff; --fg:#1b1b1f; --card:#f2f2f5; --bad:#b3261e; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16161a; --fg:#e8e8ec; --card:#24242b; --bad:#ff8a80; }} }}
body {{ margin:0; padding:16px; background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; }}
.card {{ margin:0; padding:10px; background:var(--card); border-radius:8px; }}
.card img {{ width:100%; height:auto; border-radius:4px; display:block; }}
.card figcaption {{ margin-top:8px; word-break:break-word; }} .bad {{ color:var(--bad); font-weight:600; }}
pre {{ overflow-x:auto; }}
</style></head><body>
<h1>Comparação de custo-benefício</h1>
<p>Tamanho {html.escape(data['size'])} · {data['samples']} amostra(s) por configuração · dê uma nota de 1 a 5 em
<code>{RATINGS_FILE}</code> e rode <code>python -m openai_images rank {html.escape(run_dir.name)}</code>.</p>
<pre>{html.escape(table(SUMMARY_HEADERS, summary_rows(data['records'])))}</pre>
<div class="grid">{''.join(cards)}</div></body></html>
"""
    path = run_dir / SHEET_FILE
    path.write_text(body, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------- ranking


def read_ratings(path: Path) -> dict[str, float]:
    """``arquivo -> nota`` das linhas preenchidas do ratings.csv (1 a 5)."""
    ratings: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for number, row in enumerate(csv.DictReader(fh), start=2):
            raw = (row.get("rating") or "").strip().replace(",", ".")
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                raise ValueError(f"{path.name}, linha {number}: nota inválida {raw!r}") from None
            if not 1 <= value <= 5:
                raise ValueError(f"{path.name}, linha {number}: a nota deve estar entre 1 e 5 (recebido {raw})")
            ratings[row["image"]] = value
    return ratings


def rank(run_dir: Path, min_rating: float = 4.0) -> str:
    """Ranking de custo-benefício de uma comparação já avaliada; devolve o texto (Markdown) e o grava."""
    results = run_dir / RESULTS_FILE
    if not results.is_file():
        raise FileNotFoundError(f"{results} não existe; rode a comparação antes")
    data = json.loads(results.read_text(encoding="utf-8"))
    ratings_path = run_dir / RATINGS_FILE
    if not ratings_path.is_file():
        raise FileNotFoundError(f"{ratings_path} não existe")
    ratings = read_ratings(ratings_path)
    if not ratings:
        raise ValueError(f"nenhuma nota preenchida em {ratings_path.name}: dê notas de 1 a 5 e rode de novo")

    rows = []
    for r in data["records"]:
        if r.get("error") or not r.get("images"):
            continue
        scores = [ratings[i["file"]] for i in r["images"] if i["file"] in ratings]
        if not scores:
            continue
        cost = cost_per_image(r)
        rows.append({
            "label": r["label"], "rating": statistics.mean(scores), "n": len(scores), "cost": cost,
            "height_ok": all(i.get("meets_height") is not False for i in r["images"]),
            "estimated": r.get("cost_per_image") is None,
        })
    if not rows:
        raise ValueError("nenhuma imagem avaliada corresponde às configurações do results.json")

    eligible = [x for x in rows if x["height_ok"] and x["cost"] is not None]
    acceptable = sorted((x for x in eligible if x["rating"] >= min_rating), key=lambda x: (x["cost"], -x["rating"]))
    winner = acceptable[0] if acceptable else None
    best = max(eligible, key=lambda x: (x["rating"], -x["cost"]), default=None)

    table_rows = []
    for x in sorted(rows, key=lambda x: (x["cost"] is None, x["cost"] or 0)):
        flags = []
        if not x["height_ok"]:
            flags.append("altura abaixo do mínimo")
        if x is winner:
            flags.append("MELHOR CUSTO-BENEFÍCIO")
        table_rows.append([
            x["label"], f"{x['rating']:.2f}", str(x["n"]),
            usd(x["cost"]) + (" (est.)" if x["estimated"] else ""),
            f"{x['rating'] / x['cost']:.0f}" if x["cost"] else "—", ", ".join(flags),
        ])
    lines = ["# Ranking de custo-benefício", "",
             f"Regra: a combinação mais barata com nota média >= {min_rating:g} e altura >= o mínimo do projeto.", "",
             "```", table(["config", "nota média", "imagens", "custo/img", "nota por US$", "observação"], table_rows),
             "```", ""]
    if winner:
        lines.append(f"**Melhor custo-benefício:** `{winner['label']}` — nota {winner['rating']:.2f} "
                     f"por {usd(winner['cost'])} a imagem.")
        if best and best is not winner and best["cost"] and winner["cost"]:
            extra = best["cost"] - winner["cost"]
            lines.append(f"Subir para `{best['label']}` (nota {best['rating']:.2f}) custa "
                         f"{usd(extra)} a mais por imagem ({best['cost'] / winner['cost'] - 1:+.0%}) "
                         f"por {best['rating'] - winner['rating']:+.2f} de nota.")
    else:
        lines.append(f"Nenhuma combinação elegível atingiu nota {min_rating:g}. "
                     + (f"A de maior nota foi `{best['label']}` ({best['rating']:.2f}); "
                        "reduza --min-rating ou teste configurações de qualidade maior." if best else ""))
    text = "\n".join(lines) + "\n"
    (run_dir / RANKING_FILE).write_text(text, encoding="utf-8")
    return text
