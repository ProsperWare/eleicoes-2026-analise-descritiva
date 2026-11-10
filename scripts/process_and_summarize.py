#!/usr/bin/env python3
"""
Residual do linear (T1->T2) + top 2% urnas outliers.
"""

from pathlib import Path
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np

PROCESSED = Path("/home/alexa/eleicoes-data/processed")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)

REGIOES = {
    "Norte": ["AC", "AM", "AP", "PA", "RO", "RR", "TO"],
    "Nordeste": ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
    "Centro-Oeste": ["DF", "GO", "MS", "MT"],
    "Sudeste": ["ES", "MG", "RJ", "SP"],
    "Sul": ["PR", "RS", "SC"],
}

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

        # Identificadores da urna
        id_cols = [c for c in ["NR_ZONA", "NR_SECAO", "QT_COMPARECIMENTO_t2", "QT_APTOS_t2"] if c in test_clean.columns]
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

def main():
    join_files = sorted(PROCESSED.glob("join_*.parquet"))
    print(f"Joins pré-processados encontrados: {len(join_files)}")
    if not join_files:
        (RESULTS / "status.txt").write_text("Nenhum join pré-processado encontrado\n", encoding="utf-8")
        return

    rows = []
    all_urnas = []

    for jf in join_files:
        uf = jf.stem.replace("join_", "")
        print(f"Processando {uf}...")
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
            all_urnas.append(res_df)
            print(f"  n={info['n']} | dentro ±5pp={info['pct_dentro_5']}%")

    if rows:
        with open(RESULTS / "residual_resumo.txt", "w", encoding="utf-8") as f:
            f.write("Residual do linear (teste)\n")
            f.write(f"{'UF':<6} {'N':>8} {'ResMed':>8} {'ResStd':>8} {'%<=5pp':>8}\n")
            f.write("-"*40 + "\n")
            for r in rows:
                f.write(f"{r['uf']:<6} {r['n']:>8} {r['res_medio']:>8} {r['res_std']:>8} {r['pct_dentro_5']:>8}\n")

    if all_urnas:
        urnas = pd.concat(all_urnas, ignore_index=True)
        urnas["abs_residual"] = urnas["residual"].abs()
        n_top = max(1, int(len(urnas) * 0.02))
        top_outliers = urnas.nlargest(n_top, "abs_residual")
        top_outliers = top_outliers.round({"residual": 3, "previsto": 2, "observado": 2, "abs_residual": 3})
        top_outliers.to_csv(RESULTS / "top2pct_urnas_outliers.csv", index=False)
        print(f"Top 2% urnas outliers salvas ({n_top} urnas). Max residual abs: {top_outliers['abs_residual'].max():.2f} p.p.")

    status = f"UFs processadas: {len(rows)}\nTop 2% gerado\n"
    (RESULTS / "status.txt").write_text(status, encoding="utf-8")
    print(status)

if __name__ == "__main__":
    main()
