#!/usr/bin/env python3
"""
Orquestrador: decide quais scripts de download e processamento executar.

Uso:
  python scripts/orquestrar.py --anos 2026 2022 2018
  python scripts/orquestrar.py --anos 2022 --skip-download
  python scripts/orquestrar.py --anos 2026 --ufs SP MG
  python scripts/orquestrar.py --anos 2026 2022 2018 --skip-benchmark
"""

import argparse
import functools
import subprocess
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
print = functools.partial(print, flush=True)

def run(cmd: list[str], desc: str):
    print(f"\n=== {desc} ===")
    print(" ".join(cmd))
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"Aviso: {desc} retornou código {result.returncode}")
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Orquestra downloads e processamentos por ano")
    parser.add_argument("--anos", nargs="+", default=["2026"], help="Anos a processar (ex: 2026 2022 2018)")
    parser.add_argument("--ufs", nargs="*", default=[], help="UFs específicas (opcional)")
    parser.add_argument("--skip-download", action="store_true", help="Pula todos os downloads")
    parser.add_argument("--skip-preprocess", action="store_true", help="Pula pré-processamento")
    parser.add_argument("--skip-residual", action="store_true", help="Pula cálculo de residual")
    parser.add_argument("--skip-socio", action="store_true", help="Pula cruzamento socioeconômico (IDH + transferências)")
    parser.add_argument("--skip-benchmark", action="store_true", help="Pula benchmark de modelos vs logit")
    args = parser.parse_args()

    anos = [a.strip() for a in args.anos]
    print(f"Orquestrador iniciado. Anos: {anos}")

    # Downloads
    if not args.skip_download:
        if "2026" in anos:
            cmd = [sys.executable, "scripts/download_bu.py"]
            if args.ufs:
                cmd += ["--ufs"] + args.ufs
            else:
                cmd += ["--all"]
            run(cmd, "Download BUs 2026")

        if "2018" in anos:
            run([sys.executable, "scripts/download_bu_2018.py"], "Download BUs 2018")

        if "2022" in anos:
            run([sys.executable, "scripts/download_bu_2022.py"], "Download BUs 2022")

        if "2026" in anos:
            run([sys.executable, "scripts/download_logs.py"], "Download logs 2026")

        # Bases socioeconômicas (IDH + transferências)
        run([sys.executable, "scripts/baixar_socioeconomico.py"], "Download bases socioeconômicas (IDH + transferências)")

    # Diagnóstico
    if "2018" in anos or "2022" in anos:
        run([sys.executable, "scripts/diagnostico_csv.py"], "Diagnóstico CSVs 2018/2022")

    # Pré-processamento
    if not args.skip_preprocess:
        for ano in anos:
            if ano in ("2026", "2022", "2018"):
                run([sys.executable, "scripts/preprocess_bu.py", "--ano", ano], f"Pré-processar BUs {ano}")

    # Residual por ano + comparação
    if not args.skip_residual:
        run([sys.executable, "scripts/process_and_summarize.py", "--anos"] + anos, "Residual + top 2% por ano")
        if "2026" in anos:
            run([sys.executable, "scripts/residual_fatores.py"], "Fatores do residual 2026")
        if set(["2018", "2022", "2026"]).issubset(set(anos)):
            run([sys.executable, "scripts/intersecao_3anos.py"], "Interseção top 2% dos 3 anos")
            run([sys.executable, "scripts/detalhar_intersecao.py"], "Detalhar interseção")
            run([sys.executable, "scripts/top_positivos_negativos.py"], "Top positivos e negativos")
            run([sys.executable, "scripts/ac_zona3_anos.py"], "Comparação zona 3 AC nos 3 anos")

    # Cruzamento socioeconômico (IDH + transferências federais)
    if not args.skip_socio:
        for ano in anos:
            if ano in ("2026", "2022", "2018"):
                run([sys.executable, "scripts/cruzar_socioeconomico.py", "--ano", ano], f"Cruzamento socioeconômico {ano}")

    # Benchmark de ML. Roda por último: um erro aqui não interrompe o que veio antes,
    # e o subprocess segue mesmo com código diferente de zero.
    if not args.skip_benchmark:
        anos_ml = [a for a in anos if a in ("2026", "2022", "2018")]
        if anos_ml:
            run(
                [sys.executable, "scripts/benchmark_modelos.py", "--anos", *anos_ml],
                "Benchmark de modelos vs logit",
            )

    print("\nOrquestração concluída.")

if __name__ == "__main__":
    main()
