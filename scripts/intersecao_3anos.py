#!/usr/bin/env python3
"""
Calcula seções que estão no top 2% de residual em 2018, 2022 e 2026.
Grava resultados/intersecao_3anos.txt e .csv
"""

from pathlib import Path
import pandas as pd

RESULTS = Path("resultados")
KEYS = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]

def load(ano):
    p = RESULTS / f"top2pct_urnas_{ano}.csv"
    if not p.exists():
        print(f"Não encontrado: {p}")
        return None
    df = pd.read_csv(p)
    # Normaliza chaves
    for k in KEYS:
        if k in df.columns:
            df[k] = df[k].astype(str).str.strip()
    return df

def main():
    d2018 = load("2018")
    d2022 = load("2022")
    d2026 = load("2026")
    if any(x is None for x in (d2018, d2022, d2026)):
        return

    common = [k for k in KEYS if k in d2018.columns and k in d2022.columns and k in d2026.columns]
    if not common:
        print("Chaves comuns não encontradas")
        return

    m = d2026.merge(d2022, on=common, how="inner", suffixes=("_2026", "_2022"))
    m = m.merge(d2018, on=common, how="inner", suffixes=("", "_2018"))

    out_csv = RESULTS / "intersecao_3anos.csv"
    m.to_csv(out_csv, index=False)

    lines = [
        f"Seções no top 2% nos três anos: {len(m)}",
        f"Chaves usadas: {common}",
        "",
        "Primeiras 30:",
    ]
    show_cols = common + [c for c in m.columns if "residual" in c.lower() or "uf" in c.lower()]
    show_cols = [c for c in show_cols if c in m.columns][:12]
    lines.append(m[show_cols].head(30).to_string(index=False))

    out_txt = RESULTS / "intersecao_3anos.txt"
    out_txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
