#!/usr/bin/env python3
"""
Processa Boletins de Urna baixados (zips em data/raw/),
agrega por seção, extrai percentuais de candidatos principais,
junt a T1 x T2 e calcula baseline de transferência proporcional.
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

def extract_candidate_pcts(df: pd.DataFrame, keys: list, cargo_code: str = None, top_n: int = 3) -> pd.DataFrame:
    """Extrai percentuais dos principais candidatos por seção."""
    if "QT_VOTOS" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["QT_VOTOS"] = to_numeric_safe(work["QT_VOTOS"])

    # Filtra por cargo se informado (1 = presidente)
    if cargo_code and "CD_CARGO" in work.columns:
        work = work[work["CD_CARGO"] == cargo_code]

    if work.empty:
        return pd.DataFrame()

    # Total de votos por seção
    totals = work.groupby(keys)["QT_VOTOS"].sum().reset_index().rename(columns={"QT_VOTOS": "TOTAL_VOTOS"})

    # Top candidatos por volume total
    cand_totals = work.groupby(["NR_VOTAVEL", "NM_VOTAVEL"])["QT_VOTOS"].sum().reset_index()
    top_cands = cand_totals.nlargest(top_n, "QT_VOTOS")["NR_VOTAVEL"].tolist()

    result = totals.copy()
    for nr in top_cands:
        name = cand_totals.loc[cand_totals["NR_VOTAVEL"] == nr, "NM_VOTAVEL"].iloc[0]
        safe_name = "".join(c if c.isalnum() else "_" for c in str(name)[:30])
        col = f"PCT_{safe_name}"
        cand_votes = work[work["NR_VOTAVEL"] == nr].groupby(keys)["QT_VOTOS"].sum().reset_index()
        cand_votes = cand_votes.merge(totals, on=keys, how="left")
        cand_votes[col] = (cand_votes["QT_VOTOS"] / cand_votes["TOTAL_VOTOS"].replace(0, pd.NA)) * 100
        result = result.merge(cand_votes[keys + [col]], on=keys, how="left")

    return result

def aggregate_section(df: pd.DataFrame, label: str) -> pd.DataFrame:
    keys = [c for c in SECTION_KEYS if c in df.columns]
    if not keys:
        raise ValueError(f"Chaves de seção não encontradas em {label}")

    agg = df.groupby(keys, as_index=False).size().rename(columns={"size": "linhas"})

    for col in ["QT_COMPARECIMENTO", "QT_APTOS", "QT_ABSTENCOES"]:
        if col in df.columns:
            tmp = df.groupby(keys)[col].first().reset_index()
            tmp[col] = to_numeric_safe(tmp[col])
            agg = agg.merge(tmp, on=keys, how="left")

    # Extrai percentuais (presidente no T2, ou top no T1)
    is_t2 = "2t" in label.lower()
    pcts = extract_candidate_pcts(df, keys, cargo_code="1" if is_t2 else None, top_n=2 if is_t2 else 3)
    if not pcts.empty:
        agg = agg.merge(pcts, on=keys, how="left")

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
                "colunas": list(sec.columns),
            })
            print(f"  -> {len(raw)} linhas brutas → {len(sec)} seções")
        except Exception as e:
            print(f"  ERRO: {e}")
            reports.append({"arquivo": zp.name, "erro": str(e)})

    with open(RESULTS / "resumo_secoes.txt", "w", encoding="utf-8") as f:
        f.write("Agregação por seção + percentuais\n" + "="*40 + "\n\n")
        for r in reports:
            f.write(str(r) + "\n\n")

    # Join T1 x T2 e baseline
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

        # Baseline: transferência proporcional de comparecimento
        if "QT_COMPARECIMENTO_t1" in merged.columns and "QT_COMPARECIMENTO_t2" in merged.columns:
            merged["diff_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] - merged["QT_COMPARECIMENTO_t1"]
            merged["razao_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] / merged["QT_COMPARECIMENTO_t1"].replace(0, pd.NA)

        summary = {
            "uf": uf,
            "secoes_join": len(merged),
            "chaves": keys,
        }
        if "diff_comparecimento" in merged.columns:
            summary["diff_comp_media"] = float(merged["diff_comparecimento"].mean())
            summary["diff_comp_mediana"] = float(merged["diff_comparecimento"].median())

        # Lista colunas de percentual encontradas
        pct_cols = [c for c in merged.columns if c.startswith("PCT_")]
        summary["colunas_pct"] = pct_cols

        with open(RESULTS / f"join_{uf}_resumo.txt", "w", encoding="utf-8") as f:
            f.write(str(summary) + "\n")

        # Amostra com percentuais
        sample_cols = keys + [c for c in merged.columns if "PCT_" in c or "COMPARECIMENTO" in c or "diff_" in c]
        merged[sample_cols].head(10).to_csv(RESULTS / f"amostra_secao_{uf}.csv", index=False, sep=";")

    print(f"\nResultados em {RESULTS.resolve()}")

if __name__ == "__main__":
    main()
