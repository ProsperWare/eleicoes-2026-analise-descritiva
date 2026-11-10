#!/usr/bin/env python3
"""
Modelo de árvore para entender o peso dos fatores no residual.
"""

from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

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

def get_base_features(merged: pd.DataFrame) -> list:
    features = [c for c in merged.columns if c.startswith("PCT_") and c.endswith("_t1")]
    if not features:
        features = [c for c in merged.columns if c.startswith("PCT_") and not c.endswith("_t2") and "T2_" not in c]
    return features

def split_mult3(merged: pd.DataFrame):
    if "NR_SECAO" not in merged.columns:
        return None, None
    work = merged.copy()
    work["NR_SECAO_num"] = to_numeric_safe(work["NR_SECAO"])
    train = work[work["NR_SECAO_num"] % 3 == 0]
    test = work[work["NR_SECAO_num"] % 3 != 0]
    return train, test

def main():
    join_files = sorted(PROCESSED.glob("join_*.parquet"))
    print(f"Joins encontrados: {len(join_files)}")

    all_data = []

    for jf in join_files:
        uf = jf.stem.replace("join_", "")
        print(f"Processando {uf}...")
        try:
            merged = pd.read_parquet(jf)
        except Exception:
            continue
        if len(merged) == 0:
            continue

        col_t1, col_t2 = find_best_candidate_match(merged)
        if not col_t1:
            continue

        train, test = split_mult3(merged)
        if train is None or len(train) < 50:
            continue

        base_features = get_base_features(merged)
        train_clean = train.dropna(subset=[col_t2] + base_features)
        test_clean = test.dropna(subset=[col_t2] + base_features)
        if len(train_clean) < 50:
            continue

        # Modelo base T1 -> T2
        X_tr = train_clean[base_features]
        y_tr = train_clean[col_t2]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        from sklearn.linear_model import LinearRegression
        lin = LinearRegression()
        lin.fit(X_tr_s, y_tr)

        X_te = test_clean[base_features]
        y_te = test_clean[col_t2]
        pred = lin.predict(scaler.transform(X_te))
        residual = y_te - pred

        # Fatores para explicar o residual
        fatores = []
        for col in ["QT_COMPARECIMENTO_t2", "QT_APTOS_t2", "QT_COMPARECIMENTO_t1", "QT_APTOS_t1"]:
            if col in test_clean.columns:
                fatores.append(col)
        # Abstenção relativa
        if "QT_COMPARECIMENTO_t2" in test_clean.columns and "QT_APTOS_t2" in test_clean.columns:
            test_clean = test_clean.copy()
            test_clean["abstencao_rel"] = 1 - (test_clean["QT_COMPARECIMENTO_t2"] / test_clean["QT_APTOS_t2"].replace(0, np.nan))
            fatores.append("abstencao_rel")

        if not fatores:
            continue

        df_res = test_clean[fatores].copy()
        df_res["residual"] = residual.values
        df_res["uf"] = uf
        df_res = df_res.dropna()
        if len(df_res) > 0:
            all_data.append(df_res)

    if not all_data:
        print("Sem dados para modelar residual.")
        return

    full = pd.concat(all_data, ignore_index=True)
    print(f"Total de urnas para residual: {len(full)}")

    # Árvore no residual
    fatores_final = [c for c in full.columns if c not in ["residual", "uf"]]
    X = full[fatores_final].fillna(0)
    y = full["residual"]

    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=100, random_state=42)
    tree.fit(X, y)

    # Importância
    imp = pd.DataFrame({
        "fator": fatores_final,
        "importancia": tree.feature_importances_.round(4)
    }).sort_values("importancia", ascending=False)

    imp.to_csv(RESULTS / "residual_fatores_importancia.csv", index=False)

    with open(RESULTS / "residual_fatores.txt", "w", encoding="utf-8") as f:
        f.write("Peso dos fatores no residual (Decision Tree)\n")
        f.write("="*50 + "\n")
        for _, row in imp.iterrows():
            f.write(f"{row['fator']:<30} {row['importancia']:.4f}\n")
        f.write(f"\nR2 da árvore: {tree.score(X, y):.4f}\n")
        f.write(f"N urnas: {len(full)}\n")

    print(imp.to_string(index=False))
    print(f"R2: {tree.score(X, y):.4f}")
    print(f"Resultados em {RESULTS.resolve()}")

if __name__ == "__main__":
    main()
