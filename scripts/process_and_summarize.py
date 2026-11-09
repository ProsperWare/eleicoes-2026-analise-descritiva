#!/usr/bin/env python3
"""
Treina modelos (árvore, linear, logit) apenas nas seções múltiplas de 3
e avalia no restante. Gera resumo por UF.
"""

import zipfile
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import numpy as np

RAW = Path("data/raw")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)

SECTION_KEYS = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
TOLERANCE_PP = 5.0
MAX_DEPTH = 4
BINARY_THRESHOLD = 50.0

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
    if "QT_VOTOS" not in df.columns:
        return pd.DataFrame()
    work = df.copy()
    work["QT_VOTOS"] = to_numeric_safe(work["QT_VOTOS"])
    if cargo_code and "CD_CARGO" in work.columns:
        work = work[work["CD_CARGO"] == cargo_code]
    if work.empty:
        return pd.DataFrame()
    totals = work.groupby(keys)["QT_VOTOS"].sum().reset_index().rename(columns={"QT_VOTOS": "TOTAL_VOTOS"})
    cand_totals = work.groupby(["NR_VOTAVEL", "NM_VOTAVEL"])["QT_VOTOS"].sum().reset_index()
    top_cands = cand_totals.nlargest(top_n, "QT_VOTOS")["NR_VOTAVEL"].tolist()
    result = totals.copy()
    for nr in top_cands:
        name = cand_totals.loc[cand_totals["NR_VOTAVEL"] == nr, "NM_VOTAVEL"].iloc[0]
        safe_name = "".join(c if c.isalnum() else "_" for c in str(name)[:40])
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
    is_t2 = "2t" in label.lower()
    pcts = extract_candidate_pcts(df, keys, cargo_code="1" if is_t2 else None, top_n=2 if is_t2 else 3)
    if not pcts.empty:
        agg = agg.merge(pcts, on=keys, how="left")
    agg["origem"] = label
    return agg

def find_best_candidate_match(merged: pd.DataFrame) -> tuple:
    pct_t1 = [c for c in merged.columns if c.startswith("PCT_") and c.endswith("_t1")]
    pct_t2 = [c for c in merged.columns if c.startswith("PCT_") and c.endswith("_t2")]
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

def evaluate_regression(model, X, y):
    if len(X) == 0 or len(y) == 0:
        return None, None
    pred = model.predict(X)
    pred = np.clip(pred, 0, 100)
    erro = np.abs(pred - y)
    within = (erro <= TOLERANCE_PP).sum()
    cob = 100 * within / len(y)
    return round(cob, 2), round(float(erro.mean()), 2)

def main():
    zips = sorted(RAW.glob("bweb_*.zip"))
    if not zips:
        print("Nenhum zip em data/raw/")
        return
    section_dfs = {}
    for zp in zips:
        print(f"Processando {zp.name}...")
        try:
            raw = load_bu_zip(zp)
            label = zp.stem
            sec = aggregate_section(raw, label)
            section_dfs[label] = sec
            print(f"  -> {len(raw)} linhas → {len(sec)} seções")
        except Exception as e:
            print(f"  ERRO: {e}")
    t1_keys = [k for k in section_dfs if "1t" in k]
    t2_keys = [k for k in section_dfs if "2t" in k]
    summary = []
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
        if "QT_COMPARECIMENTO_t1" in merged.columns and "QT_COMPARECIMENTO_t2" in merged.columns:
            merged["diff_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] - merged["QT_COMPARECIMENTO_t1"]
            merged["razao_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] / merged["QT_COMPARECIMENTO_t1"].replace(0, pd.NA)
        col_t1, col_t2 = find_best_candidate_match(merged)
        if not col_t1:
            continue
        train, test = split_mult3(merged)
        if train is None or len(train) < 30:
            continue
        features = get_features(merged)
        # Remove NaN no alvo e nas features
        train_clean = train.dropna(subset=[col_t2] + features)
        test_clean = test.dropna(subset=[col_t2] + features)
        if len(train_clean) < 30:
            print(f"  {uf}: poucas seções válidas no treino após dropna")
            continue
        X_train = train_clean[features]
        y_train = train_clean[col_t2]
        X_test = test_clean[features]
        y_test = test_clean[col_t2]
        # Árvore
        tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=42, min_samples_leaf=15)
        tree.fit(X_train, y_train)
        cob_tree_tr, err_tree_tr = evaluate_regression(tree, X_train, y_train)
        cob_tree_te, err_tree_te = evaluate_regression(tree, X_test, y_test)
        rules = export_text(tree, feature_names=list(features))[:2500]
        # Linear
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test) if len(X_test) > 0 else X_test
        lin = LinearRegression()
        lin.fit(X_train_s, y_train)
        cob_lin_tr, err_lin_tr = evaluate_regression(lin, X_train_s, y_train)
        cob_lin_te, err_lin_te = evaluate_regression(lin, X_test_s, y_test) if len(X_test) > 0 else (None, None)
        # Logit binário
        y_bin_train = (y_train >= BINARY_THRESHOLD).astype(int)
        logit_acc = None
        if y_bin_train.nunique() >= 2 and len(X_test) > 0:
            try:
                logit = LogisticRegression(max_iter=1000, random_state=42)
                logit.fit(X_train_s, y_bin_train)
                y_bin_test = (y_test >= BINARY_THRESHOLD).astype(int)
                pred_bin = logit.predict(X_test_s)
                logit_acc = round(100 * accuracy_score(y_bin_test, pred_bin), 2)
            except Exception:
                logit_acc = None
        # Salva regras
        with open(RESULTS / f"regras_mult3_{uf}.txt", "w", encoding="utf-8") as f:
            f.write(f"Árvore treinada nos múltiplos de 3 — {uf}\n")
            f.write("="*50 + "\n")
            f.write(rules + "\n")
        summary.append({
            "uf": uf,
            "n_treino": len(train_clean),
            "n_teste": len(test_clean),
            "arvore_treino": cob_tree_tr,
            "arvore_teste": cob_tree_te,
            "linear_treino": cob_lin_tr,
            "linear_teste": cob_lin_te,
            "logit_teste": logit_acc,
        })
    if summary:
        with open(RESULTS / "resumo_mult3.txt", "w", encoding="utf-8") as f:
            f.write("Resumo: modelos treinados apenas nos múltiplos de 3\n")
            f.write("="*70 + "\n")
            f.write(f"{'UF':<6} {'Treino':>8} {'Teste':>8} {'ArvTr%':>8} {'ArvTe%':>8} {'LinTr%':>8} {'LinTe%':>8} {'LogitTe%':>10}\n")
            f.write("-"*70 + "\n")
            for m in summary:
                f.write(f"{m['uf']:<6} {m['n_treino']:>8} {m['n_teste']:>8} {str(m['arvore_treino']):>8} {str(m['arvore_teste']):>8} {str(m['linear_treino']):>8} {str(m['linear_teste']):>8} {str(m['logit_teste']):>10}\n")
    print(f"\nResultados em {RESULTS.resolve()}")

if __name__ == "__main__":
    main()
