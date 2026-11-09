#!/usr/bin/env python3
"""
Pré-processa BUs: agrega por seção e faz join T1×T2 por UF.
Salva resultados em /home/alexa/eleicoes-data/processed/ como Parquet.
"""

import zipfile
from pathlib import Path
import pandas as pd

RAW = Path("/home/alexa/eleicoes-data/raw")
PROCESSED = Path("/home/alexa/eleicoes-data/processed")
PROCESSED.mkdir(exist_ok=True)

SECTION_KEYS = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]

DIREITA_PRES = {"22", "15"}
ESQUERDA_PRES = {"13", "12", "30"}

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

def detect_columns(df: pd.DataFrame) -> dict:
    cols = list(df.columns)
    cargo_col = None
    for c in ["CD_CARGO", "CD_CARGO_PERGUNTA", "DS_CARGO_PERGUNTA"]:
        if c in cols:
            cargo_col = c
            break
    votos_col = None
    for c in ["QT_VOTOS", "QT_VOTOS_NOMINAIS", "QT_VOTOS_VALIDOS"]:
        if c in cols:
            votos_col = c
            break
    # fallback: qualquer coluna com VOTO no nome
    if not votos_col:
        for c in cols:
            if "VOTO" in c.upper() and "QT_" in c.upper():
                votos_col = c
                break
    return {"cargo": cargo_col, "votos": votos_col}

def extract_cargo_pcts(df: pd.DataFrame, keys: list, cargo_code: str, top_n: int = 3, prefix: str = "") -> pd.DataFrame:
    detected = detect_columns(df)
    cargo_col = detected["cargo"]
    votos_col = detected["votos"]
    if not cargo_col or not votos_col:
        print(f"    [DEBUG] Colunas não detectadas. cargo={cargo_col} votos={votos_col}")
        print(f"    Colunas com CARGO/VOTO/QT: {[c for c in df.columns if any(x in c.upper() for x in ['CARGO','VOTO','QT_'])]}")
        return pd.DataFrame()
    df = df.copy()
    df[cargo_col] = df[cargo_col].astype(str).str.strip().str.lstrip("0")
    work = df[df[cargo_col] == cargo_code].copy()
    if work.empty:
        print(f"    [DEBUG] Cargo {cargo_code} vazio em {cargo_col}. Únicos: {df[cargo_col].unique()[:8]}")
        return pd.DataFrame()
    work[votos_col] = to_numeric_safe(work[votos_col])
    totals = work.groupby(keys)[votos_col].sum().reset_index().rename(columns={votos_col: "TOTAL_VOTOS"})
    cand_totals = work.groupby(["NR_VOTAVEL", "NM_VOTAVEL"])[votos_col].sum().reset_index()
    top_cands = cand_totals.nlargest(top_n, votos_col)["NR_VOTAVEL"].tolist()
    result = totals[keys].copy()
    for nr in top_cands:
        name = cand_totals.loc[cand_totals["NR_VOTAVEL"] == nr, "NM_VOTAVEL"].iloc[0]
        safe = "".join(c if c.isalnum() else "_" for c in str(name)[:30])
        col = f"PCT_{prefix}{safe}"
        cand_votes = work[work["NR_VOTAVEL"] == nr].groupby(keys)[votos_col].sum().reset_index()
        cand_votes = cand_votes.merge(totals, on=keys, how="left")
        cand_votes[col] = (cand_votes[votos_col] / cand_votes["TOTAL_VOTOS"].replace(0, pd.NA)) * 100
        result = result.merge(cand_votes[keys + [col]], on=keys, how="left")
    if cargo_code == "1":
        dir_votes = work[work["NR_VOTAVEL"].isin(DIREITA_PRES)].groupby(keys)[votos_col].sum().reset_index()
        esq_votes = work[work["NR_VOTAVEL"].isin(ESQUERDA_PRES)].groupby(keys)[votos_col].sum().reset_index()
        dir_votes = dir_votes.merge(totals, on=keys, how="left")
        esq_votes = esq_votes.merge(totals, on=keys, how="left")
        dir_votes["PCT_DIR_PRES"] = (dir_votes[votos_col] / dir_votes["TOTAL_VOTOS"].replace(0, pd.NA)) * 100
        esq_votes["PCT_ESQ_PRES"] = (esq_votes[votos_col] / esq_votes["TOTAL_VOTOS"].replace(0, pd.NA)) * 100
        result = result.merge(dir_votes[keys + ["PCT_DIR_PRES"]], on=keys, how="left")
        result = result.merge(esq_votes[keys + ["PCT_ESQ_PRES"]], on=keys, how="left")
    return result

def aggregate_section(df: pd.DataFrame, label: str) -> pd.DataFrame:
    keys = [c for c in SECTION_KEYS if c in df.columns]
    if not keys:
        raise ValueError(f"Chaves de seção não encontradas em {label}")
    detected = detect_columns(df)
    print(f"    Detectado: cargo={detected['cargo']} votos={detected['votos']}")
    agg = df.groupby(keys, as_index=False).size().rename(columns={"size": "linhas"})
    for col in ["QT_COMPARECIMENTO", "QT_APTOS", "QT_ABSTENCOES"]:
        if col in df.columns:
            tmp = df.groupby(keys)[col].first().reset_index()
            tmp[col] = to_numeric_safe(tmp[col])
            agg = agg.merge(tmp, on=keys, how="left")
    is_t2 = "2t" in label.lower()
    if is_t2:
        pcts = extract_cargo_pcts(df, keys, "1", top_n=2, prefix="T2_")
        if not pcts.empty:
            agg = agg.merge(pcts, on=keys, how="left")
            print(f"    T2: {len([c for c in pcts.columns if c.startswith('PCT_')])} colunas PCT")
        else:
            print(f"    T2: nenhuma coluna PCT gerada")
    else:
        for cargo, prefix, top in [("1", "PRES_", 3), ("3", "GOV_", 2), ("5", "SEN_", 2), ("6", "DEP_", 2)]:
            pcts = extract_cargo_pcts(df, keys, cargo, top_n=top, prefix=prefix)
            if not pcts.empty:
                agg = agg.merge(pcts, on=keys, how="left")
                print(f"    Cargo {cargo}: {len([c for c in pcts.columns if c.startswith('PCT_')])} colunas PCT")
            else:
                print(f"    Cargo {cargo}: nenhuma coluna PCT")
    agg["origem"] = label
    return agg

def needs_rebuild(zip_path: Path, parquet_path: Path) -> bool:
    if not parquet_path.exists():
        return True
    return zip_path.stat().st_mtime > parquet_path.stat().st_mtime

def force_suffix(df: pd.DataFrame, suffix: str, keys: list) -> pd.DataFrame:
    rename = {}
    for c in df.columns:
        if c not in keys and not c.endswith(suffix):
            rename[c] = f"{c}{suffix}"
    return df.rename(columns=rename)

def main():
    zips = sorted(RAW.glob("bweb_*.zip"))
    print(f"Zips encontrados: {len(zips)}")
    if not zips:
        print("Nenhum zip")
        return

    section_dfs = {}
    rebuilt = 0
    for zp in zips:
        label = zp.stem
        parquet_path = PROCESSED / f"{label}.parquet"
        if needs_rebuild(zp, parquet_path):
            print(f"Reprocessando {zp.name}...")
            try:
                raw = load_bu_zip(zp)
                sec = aggregate_section(raw, label)
                sec.to_parquet(parquet_path, index=False)
                section_dfs[label] = sec
                rebuilt += 1
                print(f"  -> {len(sec)} seções salvas | colunas: {len(sec.columns)}")
            except Exception as e:
                print(f"  ERRO: {e}")
        else:
            print(f"Cache válido: {label}")
            try:
                section_dfs[label] = pd.read_parquet(parquet_path)
            except Exception as e:
                print(f"  Erro ao ler cache: {e}")

    # Join T1 x T2 por UF e salva
    t1_keys = [k for k in section_dfs if "1t" in k]
    t2_keys = [k for k in section_dfs if "2t" in k]
    print(f"\nJoins: {len(t1_keys)} T1")
    for t1k in t1_keys:
        uf = t1k.split("_")[2] if "_" in t1k else "?"
        t2k = next((k for k in t2_keys if uf in k), None)
        if not t2k:
            continue
        join_path = PROCESSED / f"join_{uf}.parquet"
        df1 = section_dfs[t1k]
        df2 = section_dfs[t2k]
        keys = [c for c in SECTION_KEYS if c in df1.columns and c in df2.columns]
        if not keys:
            continue
        df1_s = force_suffix(df1, "_t1", keys)
        df2_s = force_suffix(df2, "_t2", keys)
        merged = df1_s.merge(df2_s, on=keys, how="inner")
        if "QT_COMPARECIMENTO_t1" in merged.columns and "QT_COMPARECIMENTO_t2" in merged.columns:
            merged["diff_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] - merged["QT_COMPARECIMENTO_t1"]
            merged["razao_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] / merged["QT_COMPARECIMENTO_t1"].replace(0, pd.NA)
        merged.to_parquet(join_path, index=False)
        pct_cols = [c for c in merged.columns if c.startswith("PCT_")]
        print(f"  Join {uf}: {len(merged)} seções | {len(pct_cols)} colunas PCT")

    print(f"\nPré-processamento concluído. Reconstruídos: {rebuilt} arquivos.")
    print(f"Dados em {PROCESSED}")

if __name__ == "__main__":
    main()
