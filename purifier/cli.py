"""Uso em linha de comando (lote / sem servidor).

    python -m purifier.cli processo.pdf [-o saida.md] [--anon] [--llm] [--vision]
                           [--inicio N] [--fim N] [--senha X] [--operador "Nome"]
"""

import argparse
import logging
import sys
import threading
from pathlib import Path

from .pipeline import InputError, run, shutdown_executor
from .progress import Tracker


def main() -> int:
    ap = argparse.ArgumentParser(description="Extrai texto de PDF jurídico com relatório de saúde da leitura.")
    ap.add_argument("pdf", nargs="+", help="arquivo(s) PDF")
    ap.add_argument("-o", "--saida", help="arquivo .md de saída (apenas com 1 PDF) ou pasta")
    ap.add_argument("--anon", action="store_true", help="anonimizar dados pessoais")
    ap.add_argument("--llm", action="store_true", help="corrigir erros de OCR com IA local (texto)")
    ap.add_argument("--vision", action="store_true", help="transcrever páginas comprometidas com IA de visão")
    ap.add_argument("--manter-cabecalhos", action="store_true", help="não remover cabeçalhos/rodapés repetidos")
    ap.add_argument("--inicio", type=int)
    ap.add_argument("--fim", type=int)
    ap.add_argument("--senha", default="")
    ap.add_argument("--operador", default="")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    options = {"anon": args.anon, "use_llm": args.llm, "use_vision": args.vision,
               "page_start": args.inicio, "page_end": args.fim, "password": args.senha,
               "operator": args.operador, "remove_repeated": not args.manter_cabecalhos}

    def ticker(tracker: Tracker, stop: threading.Event) -> None:
        # linha de status com estimativa de tempo, atualizada a cada segundo
        while not stop.wait(1.0):
            sys.stderr.write("\r" + tracker.line()[:110].ljust(110))
            sys.stderr.flush()

    status = 0
    try:
        for pdf in args.pdf:
            path = Path(pdf)
            try:
                tracker, stop = Tracker(options), threading.Event()
                threading.Thread(target=ticker, args=(tracker, stop), daemon=True).start()
                try:
                    markdown, name, summary = run(str(path), path.name, options, tracker)
                finally:
                    stop.set()
            except InputError as exc:
                sys.stderr.write(f"\n{path.name}: {exc}\n")
                status = 2
                continue
            if args.saida and len(args.pdf) == 1 and args.saida.lower().endswith(".md"):
                out = Path(args.saida)
            else:
                out = (Path(args.saida) if args.saida else path.parent) / name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(markdown, encoding="utf-8")
            sys.stderr.write(f"\n{path.name}: {summary['pages']} pág., saúde média {summary['avg_health']}%, "
                             f"comprometidas: {len(summary['compromised'])} -> {out}\n")
    finally:
        shutdown_executor()
    return status


if __name__ == "__main__":
    raise SystemExit(main())
