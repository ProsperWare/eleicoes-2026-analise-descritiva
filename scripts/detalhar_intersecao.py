#!/usr/bin/env python3
"""
Detalha as seções da interseção dos 3 anos com nome de município e tamanho da urna.
"""

from pathlib import Path
import pandas as pd

RESULTS = Path("resultados")
PROCESSED = Path("/home/alexa/eleicoes-data/processed")

def main():
    inter = pd.read_csv(RESULTS / "intersecao_3anos.csv")
    if inter.empty:
        print("Interseção vazia")
        return

    # Tenta pegar nome de município de um join 2026
    mun_map = {}
    for p in PROCESSED.glob("join_*.parquet"):
        if "_2018_" in p.stem or "_2022_" in p.stem:
            continue
        try:
            df = pd.read_parquet(p, columns=["CD_MUNICIPIO", "NM_MUNICIPIO"])
            for _, r in df.drop_duplicates("CD_MUNICIPIO").iterrows():
                mun_map[str(r["CD_MUNICIPIO"]).strip()] = r["NM_MUNICIPIO"]
        except Exception:
            continue

    inter["CD_MUNICIPIO"] = inter["CD_MUNICIPIO"].astype(str).str.strip()
    inter["municipio"] = inter["CD_MUNICIPIO"].map(mun_map).fillna("?")

    # Tamanho (comparecimento T2 2026)
    if "QT_COMPARECIMENTO_t2_2026" in inter.columns:
        inter["votantes_t2_2026"] = inter["QT_COMPARECIMENTO_t2_2026"]
    elif "QT_COMPARECIMENTO_t2" in inter.columns:
        inter["votantes_t2_2026"] = inter["QT_COMPARECIMENTO_t2"]

    cols = ["SG_UF_2026", "municipio", "NR_ZONA", "NR_SECAO",
            "residual_2026", "residual_2022", "residual",
            "votantes_t2_2026"]
    cols = [c for c in cols if c in inter.columns]
    out = inter[cols].copy()
    out = out.sort_values("SG_UF_2026")

    txt = RESULTS / "intersecao_3anos_detalhada.txt"
    lines = ["=== 17 seções no top 2% em 2018, 2022 e 2026 ===\n"]
    lines.append(out.to_string(index=False))
    lines.append("\n\nResumo por UF:")
    if "SG_UF_2026" in out.columns:
        resumo = out.groupby("SG_UF_2026").size().sort_values(ascending=False)
        lines.append(resumo.to_string())

    txt.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
