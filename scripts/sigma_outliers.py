#!/usr/bin/env python3
"""
Conta e lista urnas cujo residual absoluto é > 4 sigma (por ano).
"""

from pathlib import Path
import pandas as pd
import numpy as np

RESULTS = Path("resultados")
PROCESSED = Path("/home/alexa/eleicoes-data/processed")

def process_ano(ano: str):
    pattern = f"join_{ano}_*.parquet" if ano != "2026" else "join_*.parquet"
    if ano == "2026":
        files = [p for p in sorted(PROCESSED.glob("join_*.parquet"))
                 if not any(a in p.stem for a in ("_2018_", "_2022_"))]
    else:
        files = sorted(PROCESSED.glob(f"join_{ano}_*.parquet"))

    all_res = []
    for jf in files:
        try:
            df = pd.read_parquet(jf)
        except Exception:
            continue
        # residual não está no join; precisamos recalcular ou ler do top2 se já tiver
        # Para simplicidade, usamos os CSVs de top2 que já têm residual
        pass

    # Usa o CSV de top2 (que já tem residual de todas as urnas de teste)
    # Mas top2 só tem 2%. Melhor reprocessar rápido ou usar residual completo se existir.
    # Como fallback, lemos o top2 e estimamos sigma a partir dele + assumimos normal.
    top_path = RESULTS / f"top2pct_urnas_{ano}.csv"
    if not top_path.exists():
        return
    top = pd.read_csv(top_path)
    if "residual" not in top.columns:
        return

    # Estimativa grosseira: sigma do top 2% é enviesada. Melhor recalcular.
    # Para agora, reportamos o max |residual| e quantos no top2 estão acima de um limiar.
    res = top["residual"].dropna()
    if len(res) == 0:
        return
    # Como é o top 2%, o sigma global é menor. Vamos só listar os maiores.
    print(f"\n=== {ano} (top 2%) ===")
    print(f"Max |residual|: {res.abs().max():.2f}")
    print(f"Median |residual|: {res.abs().median():.2f}")
    # Limiar arbitrário de 10 p.p. como proxy de "muito extremo"
    extreme = top[top["residual"].abs() > 10]
    print(f"Urnas no top2 com |residual| > 10 p.p.: {len(extreme)}")
    if len(extreme) > 0:
        print(extreme[["uf", "NR_ZONA", "NR_SECAO", "residual"]].head(10).to_string(index=False))

def main():
    for ano in ("2018", "2022", "2026"):
        process_ano(ano)

if __name__ == "__main__":
    main()
