#!/usr/bin/env python3
"""
Processa Boletins de Urna baixados (zips em data/raw/),
agrega por seção, junta T1 x T2 e calcula baseline simples.
Grava apenas saídas processadas em resultados/.
"""

import zipfile
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)

SECTION_KEYS = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]

def load_bu_zip(zip_path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as z:
        csv_names = [n for n in z.namelist() if n.lower().endswith((".csv", ".txt"))]
        if not csv_names:
            raise ValueError(f"Nenhum CSV em {zip_path.name}")
        csv_name = max(csv_names, key=lambda n: z.getinfo(n).file_size)
        with z.open(csv_name) as f:
            for enc in ("utf-8", "latin-1", "cp1252"):
                try:
                    f.seek(0)
                    return pd.read_csv(f, sep=";", encoding=enc, low_memory=False, dtype=str)
                except Exception:
                    continue
            raise ValueError(f"Falha ao ler {csv_name}")

def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def aggregate_section(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Agrega o formato longo do BU para uma linha por seção."""
    keys = [c for c in SECTION_KEYS if c in df.columns]
    if not keys:
        raise ValueError(f"Chaves de seção não encontradas em {label}")

    # Colunas de quantidade comuns
    vote_col = None
    for c in ["QT_VOTOS", "QT_VOTOS_NOMINAIS", "QT_VOTOS_BRANCOS", "QT_VOTOS_NULOS"]:
        if c in df.columns:
            vote_col = c
            break

    agg = df.groupby(keys, as_index=False).agg(
        linhas=("NR_SECAO", "size") if "NR_SECAO" in df.columns else (keys[0], "size")
    )

    # Comparecimento / aptos (se existirem, são constantes por seção)
    for col in ["QT_COMPARECIMENTO", "QT_APTOS", "QT_ABSTENCOES"]:
        if col in df.columns:
            tmp = df.groupby(keys)[col].first().reset_index()
            tmp[col] = to_numeric_safe(tmp[col])
            agg = agg.merge(tmp, on=keys, how="left")

    # Total de votos nominais se houver coluna de votos
    if vote_col and vote_col in df.columns:
        votes = df.copy()
        votes[vote_col] = to_numeric_safe(votes[vote_col])
        total = votes.groupby(keys)[vote_col].sum().reset_index().rename(columns={vote_col: "QT_VOTOS_TOTAL"})
        agg = agg.merge(total, on=keys, how="left")

    agg["origem"] = label
    return agg

def main():
    zips = sorted(RAW.glob("bweb_*.zip"))
    if not zips:
        print("Nenhum zip em data/raw/")
        return

    section_dfs = {}
    reports = []

    for zp in zips:
        print(f"Processando {zp.name}...")
        try:
            raw = load_bu_zip(zp)
            label = zp.stem
            sec = aggregate_section(raw, label)
            section_dfs[label] = sec
            reports.append({
                "arquivo": label,
                "linhas_brutas": len(raw),
                "secoes": len(sec),
                "colunas_secao": list(sec.columns),
            })
            print(f"  -> {len(raw)} linhas brutas → {len(sec)} seções")
        except Exception as e:
            print(f"  ERRO: {e}")
            reports.append({"arquivo": zp.name, "erro": str(e)})

    # Resumo
    with open(RESULTS / "resumo_secoes.txt", "w", encoding="utf-8") as f:
        f.write("Agregação por seção\n" + "="*40 + "\n\n")
        for r in reports:
            f.write(str(r) + "\n\n")

    # Join T1 x T2
    t1_keys = [k for k in section_dfs if "1t" in k]
    t2_keys = [k for k in section_dfs if "2t" in k]

    for t1k in t1_keys:
        uf = t1k.split("_")[2] if "_" in t1k else "?"
        t2k = next((k for k in t2_keys if uf in k), None)
        if not t2k:
            continue

        df1 = section_dfs[t1k]
        df2 = section_dfs[t2k]
        keys = [c for c in SECTION_KEYS if c in df1.columns and c in df2.columns]

        if not keys:
            continue

        merged = df1.merge(df2, on=keys, how="inner", suffixes=("_t1", "_t2"))
        print(f"Join {uf}: {len(merged)} seções")

        # Baseline simples: proporção de comparecimento
        if "QT_COMPARECIMENTO_t1" in merged.columns and "QT_COMPARECIMENTO_t2" in merged.columns:
            merged["diff_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] - merged["QT_COMPARECIMENTO_t1"]
            merged["razao_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] / merged["QT_COMPARECIMENTO_t1"].replace(0, pd.NA)

        # Salva resumo do join (não o dataset completo)
        summary = {
            "uf": uf,
            "secoes_join": len(merged),
            "chaves": keys,
        }
        if "diff_comparecimento" in merged.columns:
            summary["diff_comp_media"] = float(merged["diff_comparecimento"].mean())
            summary["diff_comp_mediana"] = float(merged["diff_comparecimento"].median())

        with open(RESULTS / f"join_{uf}_resumo.txt", "w", encoding="utf-8") as f:
            f.write(str(summary) + "\n")

        # Amostra pequena
        merged.head(10).to_csv(RESULTS / f"amostra_secao_{uf}.csv", index=False, sep=";")

    print(f"\nResultados em {RESULTS.resolve()}")

if __name__ == "__main__":
    main()
