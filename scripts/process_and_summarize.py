#!/usr/bin/env python3
"""
Residual do linear (T1->T2) + top 2% urnas outliers por ano.
Também gera comparação de sobreposição entre anos quando possível.
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import argparse

PROCESSED = Path("/home/alexa/eleicoes-data/processed")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)

def to_numeric_safe(s):
    return pd.to_numeric(s, errors="coerce").fillna(0)

def find_best_candidate_match(merged: pd.DataFrame) -> tuple:
    pct_t1 = [c for c in merged.columns if c.startswith("PCT_") and c.endswith("_t1")]
    pct_t2 = [c for c in merged.columns if c.startswith("PCT_") and c.endswith("_t2")]
    if not pct_t1:
        pct_t1 = [c for c in merged.columns if c.startswith("PCT_") and not c.endswith("_t2") and "T2_" not in c]
    if not pct_t2:
        pct_t2 = [c for c in merged.columns if c.startswith("PCT_") and (c.endswith("_t2") or "T2_" in c)]
    if not pct_t1 or not pct_t2:
        return None, None
    best_corr = -1
    best_pair = (pct_t1[0], pct_t2[0])
    for c1 in pct_t1:
        for c2 in pct_t2:
            valid = merged[[c1, c2]].dropna()
            if len(valid) < 10:
                continue
            corr = valid[c1].corr(valid[c2])
            if pd.notna(corr) and corr > best_corr:
                best_corr = corr
                best_pair = (c1, c2)
    return best_pair

def get_features(merged: pd.DataFrame) -> list:
    features = [c for c in merged.columns if c.startswith("PCT_") and c.endswith("_t1")]
    if not features:
        features = [c for c in merged.columns if c.startswith("PCT_") and not c.endswith("_t2") and "T2_" not in c]
    for extra in ["QT_COMPARECIMENTO_t1", "QT_APTOS_t1", "diff_comparecimento", "razao_comparecimento"]:
        if extra in merged.columns:
            features.append(extra)
    return features

def split_mult3(merged: pd.DataFrame):
    if "NR_SECAO" not in merged.columns:
        return None, None
    work = merged.copy()
    work["NR_SECAO_num"] = to_numeric_safe(work["NR_SECAO"])
    train = work[work["NR_SECAO_num"] % 3 == 0]
    test = work[work["NR_SECAO_num"] % 3 != 0]
    return train, test

def residual_distribuicao(train_clean, test_clean, features, col_t2, uf):
    try:
        X_tr = train_clean[features]
        y_tr = train_clean[col_t2]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        lin = LinearRegression()
        lin.fit(X_tr_s, y_tr)

        if len(test_clean) == 0:
            return None, None, None
        X_te = test_clean[features]
        y_te = test_clean[col_t2]
        pred = lin.predict(scaler.transform(X_te))
        residual = y_te - pred

        info = {
            "n": len(residual),
            "res_medio": round(float(residual.mean()), 3),
            "res_std": round(float(residual.std()), 3),
            "pct_dentro_5": round(100 * (residual.abs() <= 5).mean(), 2),
        }

        id_cols = [c for c in ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "QT_COMPARECIMENTO_t2", "QT_APTOS_t2"] if c in test_clean.columns]
        res_df = test_clean[id_cols].copy() if id_cols else pd.DataFrame(index=test_clean.index)
        res_df["residual"] = residual.values
        res_df["previsto"] = pred
        res_df["observado"] = y_te.values
        res_df["uf"] = uf
        res_df = res_df.dropna(subset=["residual"])
        return info, None, res_df
    except Exception as e:
        print(f"  residual {uf}: {e}")
        return None, None, None

def process_ano(ano: str):
    pattern = f"join_{ano}_*.parquet" if ano != "2026" else "join_*.parquet"
    # Para 2026 os joins são join_UF.parquet; para outros join_ANO_UF.parquet
    if ano == "2026":
        join_files = [p for p in sorted(PROCESSED.glob("join_*.parquet")) if not any(a in p.stem for a in ("_2018_", "_2022_"))]
    else:
        join_files = sorted(PROCESSED.glob(f"join_{ano}_*.parquet"))

    print(f"Joins para {ano}: {len(join_files)}")
    if not join_files:
        return None

    rows = []
    all_urnas = []

    for jf in join_files:
        uf = jf.stem.replace(f"join_{ano}_", "").replace("join_", "")
        print(f"Processando {ano}_{uf}...")
        try:
            merged = pd.read_parquet(jf)
        except Exception as e:
            print(f"  Erro: {e}")
            continue
        if len(merged) == 0:
            continue

        col_t1, col_t2 = find_best_candidate_match(merged)
        if not col_t1:
            continue

        train, test = split_mult3(merged)
        if train is None or len(train) < 30:
            continue

        features = get_features(merged)
        train_clean = train.dropna(subset=[col_t2] + features)
        test_clean = test.dropna(subset=[col_t2] + features)
        if len(train_clean) < 30:
            continue

        info, _, res_df = residual_distribuicao(train_clean, test_clean, features, col_t2, uf)
        if info and res_df is not None and len(res_df) > 0:
            rows.append({"uf": uf, **info})
            res_df["ano"] = ano
            all_urnas.append(res_df)
            print(f"  n={info['n']} | dentro ±5pp={info['pct_dentro_5']}%")

    if not all_urnas:
        return None

    urnas = pd.concat(all_urnas, ignore_index=True)
    urnas["abs_residual"] = urnas["residual"].abs()
    n_top = max(1, int(len(urnas) * 0.02))
    top_outliers = urnas.nlargest(n_top, "abs_residual")
    top_outliers = top_outliers.round({"residual": 3, "previsto": 2, "observado": 2, "abs_residual": 3})
    out_csv = RESULTS / f"top2pct_urnas_{ano}.csv"
    top_outliers.to_csv(out_csv, index=False)
    print(f"Top 2% {ano}: {n_top} urnas -> {out_csv}")
    return urnas

def comparar_anos(anos):
    tops = {}
    for ano in anos:
        p = RESULTS / f"top2pct_urnas_{ano}.csv"
        if p.exists():
            tops[ano] = pd.read_csv(p)
    if len(tops) < 2:
        print("Menos de 2 anos com top 2%, sem comparação.")
        return

    # Chave de identificação da seção
    keys = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
    lines = ["=== Comparação Top 2% entre anos ===\n"]
    pares = []
    for i, a1 in enumerate(anos):
        for a2 in anos[i + 1:]:
            pares.append((str(a1), str(a2)))

    for a1, a2 in pares:
        if a1 in tops and a2 in tops:
            t1 = tops[a1]
            t2 = tops[a2]
            common_keys = [k for k in keys if k in t1.columns and k in t2.columns]
            if not common_keys:
                continue
            m = t1.merge(t2, on=common_keys, how="inner", suffixes=(f"_{a1}", f"_{a2}"))
            lines.append(f"{a1} ∩ {a2}: {len(m)} seções em comum no top 2%")
            lines.append(f"  ({len(t1)} em {a1}, {len(t2)} em {a2})\n")

    out = RESULTS / "comparacao_top2pct.txt"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anos", nargs="+", default=["2026"])
    args = parser.parse_args()

    for ano in args.anos:
        process_ano(ano)

    comparar_anos(args.anos)
    print("Processamento concluído.")

if __name__ == "__main__":
    main()
