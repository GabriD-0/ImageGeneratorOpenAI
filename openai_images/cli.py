from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from . import ROOT, batch, checks, report
from .client import ApiError, ImagesClient, get_api_key
from .compare import DEFAULT_CONFIGS, parse_configs, run_compare
from .pricing import MODELS, cost_from_usage, estimate, get_model, prompt_tokens_estimate
from .prompt import build_prompt, load_prompt

DEFAULT_SIZE = "1088x1664"   # a mesma resolução do projeto ImageGenerator: altura >= 1600 px, proporção 17:26
MIN_HEIGHT = 1600


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--prompt-file", type=Path, default=ROOT / "prompt.txt", help="padrão: prompt.txt")
    p.add_argument("--negative", choices=("append", "ignore"), default="append",
                   help="a API não tem negative prompt: append = acrescenta o bloco como 'Avoid: ...' (padrão); "
                        "ignore = descarta")
    p.add_argument("--min-height", type=int, default=MIN_HEIGHT, help="altura mínima exigida (padrão: 1600)")


def _add_image_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--size", default=DEFAULT_SIZE, help=f"LARGURAxALTURA (padrão: {DEFAULT_SIZE})")
    p.add_argument("--format", dest="output_format", choices=("png", "jpeg", "webp"), default="png")
    p.add_argument("--background", choices=("opaque", "transparent", "auto"),
                   help="padrão da API: auto. 'transparent' exige png/webp e dispensa o chroma key do prompt")
    _add_key_options(p)


def _add_key_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--key-color", default="#00B140", help="cor do fundo chroma key a conferir (padrão: #00B140)")
    p.add_argument("--no-key-check", action="store_true", help="não confere o fundo chroma key")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openai_images", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    est = sub.add_parser("estimate", help="tabela de custos, sem chamar a API")
    _add_common(est)
    est.add_argument("--size", action="append", help=f"repetível (padrão: {DEFAULT_SIZE} e 1024x1536)")
    est.add_argument("--models", help="lista separada por vírgula (padrão: todos)")
    est.add_argument("--all", action="store_true", help="mostra também o que não atende a altura mínima")

    gen = sub.add_parser("generate", help="gera imagem(ns) com uma combinação (gasta créditos)")
    _add_common(gen)
    _add_image_options(gen)
    gen.add_argument("--model", default="gpt-image-2.5-flare")
    gen.add_argument("--quality", default="medium", help="low, medium, high (e xhigh/max nos 2.5 e 2.6)")
    gen.add_argument("--n", type=int, default=1, help="imagens por chamada (1 a 10)")
    gen.add_argument("--complexity", choices=("low", "medium", "high"), help="só gpt-image-2.6-*")
    gen.add_argument("--out-dir", type=Path, default=ROOT / "outputs")
    gen.add_argument("--max-cost", type=float, default=0.50,
                     help="recusa se a estimativa passar disso em US$ (padrão: 0.50)")
    gen.add_argument("--dry-run", action="store_true", help="só mostra o pedido e o custo, sem chamar a API")

    cmp_ = sub.add_parser("compare", help="mesma cena em várias combinações, medindo o custo real (gasta créditos)")
    _add_common(cmp_)
    _add_image_options(cmp_)
    cmp_.add_argument("--configs", default=DEFAULT_CONFIGS,
                      help="modelo:qualidade separados por vírgula (padrão: 7 candidatas do 2.5 e do gpt-image-2)")
    cmp_.add_argument("--samples", type=int, default=1, help="imagens por combinação (padrão: 1)")
    cmp_.add_argument("--budget", type=float, default=1.00, help="teto de gasto em US$ (padrão: 1.00)")
    cmp_.add_argument("--out-dir", type=Path, default=ROOT / "runs")
    cmp_.add_argument("--yes", action="store_true", help="não pergunta antes de gastar")
    cmp_.add_argument("--dry-run", action="store_true", help="só mostra o plano e o custo estimado")

    bat = sub.add_parser("batch", help="Batch API: metade do preço, resultado em até 24 h (submit | status | fetch)")
    bat_sub = bat.add_subparsers(dest="batch_command", required=True)

    submit = bat_sub.add_parser("submit", help="envia N imagens de uma combinação num batch (gera cobrança)")
    _add_common(submit)
    _add_image_options(submit)
    submit.add_argument("--model", default="gpt-image-2.5-flare")
    submit.add_argument("--quality", default="medium", help="low, medium, high (e xhigh/max nos 2.5)")
    submit.add_argument("--n", type=int, default=1, help=f"quantas imagens (1 a {batch.MAX_REQUESTS}); cada uma é um pedido")
    submit.add_argument("--out-dir", type=Path, default=ROOT / "batches")
    submit.add_argument("--max-cost", type=float, default=0.50,
                        help="recusa se a estimativa do batch passar disso em US$ (padrão: 0.50)")
    submit.add_argument("--yes", action="store_true", help="não pergunta antes de enviar")
    submit.add_argument("--dry-run", action="store_true", help="só mostra o pedido e o custo, sem enviar nada")

    status = bat_sub.add_parser("status", help="andamento de um batch enviado")
    status.add_argument("batch_dir", type=Path, help="pasta criada por `batch submit` (batches/<data>)")

    fetch = bat_sub.add_parser("fetch", help="baixa as imagens de um batch pronto e calcula o custo real")
    fetch.add_argument("batch_dir", type=Path, help="pasta criada por `batch submit` (batches/<data>)")
    fetch.add_argument("--min-height", type=int, default=MIN_HEIGHT, help="altura mínima exigida (padrão: 1600)")
    _add_key_options(fetch)

    rank_ = sub.add_parser("rank", help="ranking de custo-benefício de uma comparação avaliada")
    rank_.add_argument("run_dir", type=Path, help="pasta da comparação (runs/<data>)")
    rank_.add_argument("--min-rating", type=float, default=4.0, help="nota média mínima aceitável (padrão: 4)")
    return parser


# ----------------------------------------------------------------------------- estimate


def cmd_estimate(args: argparse.Namespace) -> int:
    prompt_text = build_prompt(load_prompt(args.prompt_file), args.negative)
    tokens = prompt_tokens_estimate(prompt_text)
    names = [m.strip() for m in args.models.split(",")] if args.models else list(MODELS)
    sizes = args.size or [DEFAULT_SIZE, "1024x1536"]
    rows, skipped = [], set()
    for name in names:
        model = get_model(name)
        for quality in model.qualities:
            for size in sizes:
                try:
                    est = estimate(name, quality, size, tokens)
                except ValueError as err:
                    skipped.add(str(err))
                    continue
                height = int(size.split("x")[1])
                if height < args.min_height and not args.all:
                    skipped.add(f"{model.name}: altura máxima {height}px < {args.min_height}px")
                    continue
                rows.append((est, height >= args.min_height, model))
    rows.sort(key=lambda r: r[0].total_usd)
    print(f"Prompt: ~{tokens} tokens de entrada. Custos em US$ por imagem; altura mínima {args.min_height}px.\n")
    print(report.table(["modelo", "qualidade", "tamanho", "tokens saída", "US$/img", "US$ batch", "atende?"], [
        [e.model, e.quality, e.size, str(e.output_tokens or "tabela"), f"{e.total_usd:.4f}",
         f"{e.batch_usd:.4f}" if e.batch_usd is not None else "—", "sim" if ok else "NÃO"]
        for e, ok, _ in rows]))
    if skipped and not args.all:
        print("\nFora da tabela (use --all para ver): modelos de tamanho fixo (1.5 e mini) chegam a 1536px, "
              "abaixo do mínimo.")
    print("\nTokens de saída: fórmula da calculadora oficial. Entrada: ~4 caracteres por token. O custo real "
          "vem do campo 'usage' da resposta e pode diferir um pouco; `compare` mede isso.")
    unreferenced = [m.name for m in MODELS.values() if not m.in_reference and m.name in names]
    if unreferenced:
        print(f"Obs.: {', '.join(unreferenced)} constam no guia e na página de modelos, mas não na referência "
              "da Images API; confirme o acesso com uma chamada de teste.")
    return 0


# ----------------------------------------------------------------------------- generate


def _key_color(args: argparse.Namespace):
    return None if args.no_key_check else checks.parse_color(args.key_color)


def cmd_generate(args: argparse.Namespace) -> int:
    prompt_text = build_prompt(load_prompt(args.prompt_file), args.negative)
    if not 1 <= args.n <= 10:
        raise ValueError("--n deve estar entre 1 e 10")
    if args.complexity and not args.model.startswith("gpt-image-2.6"):
        raise ValueError("--complexity só existe nos modelos gpt-image-2.6-*")
    est = estimate(args.model, args.quality, args.size, prompt_tokens_estimate(prompt_text))
    total = est.total_usd * args.n
    print(f"{args.model} · quality={args.quality} · {args.size} · n={args.n}")
    print(f"Custo estimado: {report.usd(total)} ({est.output_tokens or 'tabela'} tokens de saída por imagem)")
    if args.dry_run:
        print(f"\nPrompt enviado ({len(prompt_text)} caracteres):\n{prompt_text}\n\n(--dry-run) nenhuma chamada foi feita.")
        return 0
    if total > args.max_cost:
        print(f"erro: estimativa {report.usd(total)} acima de --max-cost {report.usd(args.max_cost)}.", file=sys.stderr)
        return 2

    client = ImagesClient(get_api_key())
    resp = client.generate(model=args.model, prompt=prompt_text, size=args.size, quality=args.quality, n=args.n,
                           output_format=args.output_format, background=args.background,
                           complexity=args.complexity)
    cost = cost_from_usage(args.model, resp.usage)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{re.sub(r'[^A-Za-z0-9.]+', '_', args.model)}_{args.quality}_{args.size}_{datetime.now():%Y%m%d-%H%M%S}"
    for k, data in enumerate(resp.images, start=1):
        path = args.out_dir / f"{stem}_{k}.{args.output_format}"
        path.write_bytes(data)
        info = checks.analyze(data, min_height=args.min_height, key_color=_key_color(args))
        notes = []
        if info["meets_height"] is False:
            notes.append(f"altura {info['height']} < {args.min_height}")
        if info["key_ok"] is False:
            notes.append(f"só {info['key_share']:.0%} da borda é chroma key")
        print(f"  {path}  {info['width']}x{info['height']}" + (f"  ⚠ {'; '.join(notes)}" if notes else ""))
    (args.out_dir / f"{stem}.json").write_text(json.dumps({
        "model": args.model, "quality": args.quality, "size": args.size, "n": args.n, "prompt": prompt_text,
        "usage": resp.usage, "cost_usd": cost, "estimate_usd": total, "latency_s": round(resp.latency_s, 1),
        "request_id": resp.request_id, "response": resp.meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Custo real: {report.usd(cost) if cost is not None else 'indisponível (sem usage)'} "
          f"· {resp.latency_s:.0f}s · request {resp.request_id}")
    return 0


# ----------------------------------------------------------------------------- compare / rank


def cmd_compare(args: argparse.Namespace) -> int:
    prompt_text = build_prompt(load_prompt(args.prompt_file), args.negative)
    if args.samples < 1 or args.samples > 10:
        raise ValueError("--samples deve estar entre 1 e 10")
    configs = parse_configs(args.configs)
    client = None if args.dry_run else ImagesClient(get_api_key())
    return run_compare(
        configs=configs, prompt_text=prompt_text, prompt_tokens=prompt_tokens_estimate(prompt_text),
        size=args.size, samples=args.samples, out_dir=args.out_dir, budget=args.budget,
        assume_yes=args.yes, dry_run=args.dry_run, min_height=args.min_height, key_color=_key_color(args),
        output_format=args.output_format, background=args.background, client=client)


def cmd_rank(args: argparse.Namespace) -> int:
    print(report.rank(args.run_dir, args.min_rating))
    return 0


# ----------------------------------------------------------------------------- batch


def cmd_batch(args: argparse.Namespace) -> int:
    if args.batch_command == "submit":
        prompt_text = build_prompt(load_prompt(args.prompt_file), args.negative)
        if not 1 <= args.n <= batch.MAX_REQUESTS:
            raise ValueError(f"--n deve estar entre 1 e {batch.MAX_REQUESTS}")
        client = None if args.dry_run else ImagesClient(get_api_key())
        return batch.run_submit(
            client=client, model=args.model, quality=args.quality, size=args.size, count=args.n,
            prompt_text=prompt_text, prompt_tokens=prompt_tokens_estimate(prompt_text),
            output_format=args.output_format, background=args.background, out_dir=args.out_dir,
            max_cost=args.max_cost, assume_yes=args.yes, dry_run=args.dry_run)
    client = ImagesClient(get_api_key())
    if args.batch_command == "status":
        return batch.run_status(client=client, batch_dir=args.batch_dir)
    return batch.run_fetch(client=client, batch_dir=args.batch_dir, min_height=args.min_height,
                           key_color=_key_color(args))


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"estimate": cmd_estimate, "generate": cmd_generate, "compare": cmd_compare, "rank": cmd_rank,
               "batch": cmd_batch}
    try:
        return handler[args.command](args)
    except (ValueError, FileNotFoundError) as err:
        print(f"erro: {err}", file=sys.stderr)
        return 2
    except ApiError as err:
        print(f"erro da API ({err.status or err.kind}{', ' + err.code if err.code else ''}): {err}", file=sys.stderr)
        if err.request_id:
            print(f"request-id: {err.request_id}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrompido.", file=sys.stderr)
        return 130
