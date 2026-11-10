#!/usr/bin/env python3
"""Compara seções da zona 3 do Acre nos anos 2018, 2022 e 2026."""
from pathlib import Path
import pandas as pd

RESULTS = Path("resultados")

def main():
    lines = ["Comparação zona 3 AC (Sena Madureira / Manoel Urbano / Santa Rosa do Purus)\n"]
    for ano in ("2018", "2022", "2026"):
        p = RESULTS / f"top2pct_urnas_{ano}.csv"
        if not p.exists():
            lines.append(f"{ano}: arquivo não encontrado")
            continue
        try:
            df = pd.read_csv(p)
        except Exception as e:
            lines.append(f"{ano}: erro ao ler CSV: {e}")
            continue

        # Normaliza nome da coluna UF
        uf_col = None
        for c in df.columns:
            if c.upper() in ("SG_UF", "UF"):
                uf_col = c
                break
        if uf_col is None:
            lines.append(f"{ano}: coluna UF não encontrada. Colunas: {list(df.columns)}")
            continue

        zona_col = "NR_ZONA" if "NR_ZONA" in df.columns else None
        if zona_col is None:
            lines.append(f"{ano}: coluna NR_ZONA não encontrada")
            continue

        mask = (df[uf_col].astype(str).str.upper() == "AC") & (df[zona_col].astype(int) == 3)
        sub = df[mask].copy()
        if "residual" in sub.columns:
            sub = sub.sort_values("residual", key=lambda s: s.abs(), ascending=False)

        lines.append(f"\n===== {ano} — {len(sub)} seções da zona 3 no top 2% =====")
        if sub.empty:
            lines.append("(nenhuma seção da zona 3 no top 2%)")
            continue

        cols = [c for c in ["NR_SECAO", "residual", "previsto", "observado", "QT_COMPARECIMENTO_t2"] if c in sub.columns]
        lines.append(sub[cols].head(20).to_string(index=False))

    out = RESULTS / "ac_zona3_comparacao.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
