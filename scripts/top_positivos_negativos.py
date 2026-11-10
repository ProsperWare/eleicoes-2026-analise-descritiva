#!/usr/bin/env python3
"""
Lista as urnas com maior residual positivo e maior residual negativo (top 2%) por ano.
"""

from pathlib import Path
import pandas as pd

RESULTS = Path("resultados")
N = 15  # quantas de cada lado

def main():
    lines = []
    for ano in ("2018", "2022", "2026"):
        p = RESULTS / f"top2pct_urnas_{ano}.csv"
        if not p.exists():
            lines.append(f"\n{ano}: arquivo não encontrado")
            continue
        df = pd.read_csv(p)
        if "residual" not in df.columns:
            continue

        pos = df.nlargest(N, "residual")
        neg = df.nsmallest(N, "residual")

        lines.append(f"\n========== {ano} ==========")
        lines.append(f"\nTop {N} positivos (residual alto):")
        cols = [c for c in ["uf", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "residual", "QT_COMPARECIMENTO_t2"] if c in pos.columns]
        lines.append(pos[cols].to_string(index=False))

        lines.append(f"\nTop {N} negativos (residual baixo):")
        lines.append(neg[cols].to_string(index=False))

    out = RESULTS / "top_positivos_negativos.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
