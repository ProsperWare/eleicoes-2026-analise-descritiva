#!/usr/bin/env python3
"""
Esqueleto: processa BUs baixados e grava resumos/métricas em resultados/.
Não grava arquivos crus em resultados/.
"""

from pathlib import Path
import pandas as pd  # pip install pandas

RAW = Path("data/raw")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)

def main():
    # TODO: implementar parser dos CSVs dentro dos zips
    # Exemplo de saída esperada:
    summary = {
        "status": "esqueleto",
        "nota": "Resultados processados vão aqui. Arquivos crus ficam em data/raw/."
    }
    out = RESULTS / "resumo_inicial.txt"
    out.write_text(str(summary), encoding="utf-8")
    print(f"Resumo gravado em {out}")

if __name__ == "__main__":
    main()
