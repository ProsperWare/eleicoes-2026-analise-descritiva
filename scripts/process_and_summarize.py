#!/usr/bin/env python3
"""
Exploração combinatorial + Random Forest no split mult3.
Inclui perfil de seções com |residual| > 5 p.p. e distribuição por faixa de residual.
"""

from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from itertools import combinations

PROCESSED = Path("/home/alexa/eleicoes-data/processed")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)

TOLERANCE_PP = 5.0
MAX_DEPTH = 4
BINARY_THRESHOLD = 50.0
MIN_SAMPLES_LEAF = 15
MAX_SUBSET_SIZE = 5
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 8
RESIDUAL_ALTO = 5.0

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

def evaluate_regression(model, X, y):
    if len(X) == 0 or len(y) == 0:
        return None, None
    pred = model.predict(X)
    pred = np.clip(pred, 0, 100)
    erro = np.abs(pred - y)
    within = (erro <= TOLERANCE_PP).sum()
    cob = 100 * within / len(y)
    return round(cob, 2), round(float(erro.mean()), 2)

def evaluate_subset_full(features_subset, train_df, test_df, col_t2):
    if not features_subset or len(train_df) < 30:
        return None, None, None, None, None
    X_tr = train_df[features_subset]
    y_tr = train_df[col_t2]
    X_te = test_df[features_subset] if len(test_df) > 0 else pd.DataFrame()
    y_te = test_df[col_t2] if len(test_df) > 0 else pd.Series(dtype=float)
    if len(X_tr) < 30:
        return None, None, None, None, None

    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=42, min_samples_leaf=MIN_SAMPLES_LEAF)
    tree.fit(X_tr, y_tr)
    cob_tr, _ = evaluate_regression(tree, X_tr, y_tr)
    cob_te, _ = evaluate_regression(tree, X_te, y_te) if len(X_te) > 0 else (None, None)

    logit_acc = None
    y_bin_tr = (y_tr >= BINARY_THRESHOLD).astype(int)
    if y_bin_tr.nunique() >= 2 and len(X_te) > 0:
        try:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
            logit = LogisticRegression(max_iter=1000, random_state=42)
            logit.fit(X_tr_s, y_bin_tr)
            y_bin_te = (y_te >= BINARY_THRESHOLD).astype(int)
            pred_bin = logit.predict(X_te_s)
            logit_acc = round(100 * accuracy_score(y_bin_te, pred_bin), 2)
        except Exception:
            logit_acc = None

    return cob_te, cob_tr, tree, logit_acc, features_subset

def explore_combinations(all_features, train_clean, test_clean, col_t2):
    if not all_features:
        return None, [], None

    _, _, _, baseline_logit, _ = evaluate_subset_full(all_features, train_clean, test_clean, col_t2)

    best_cob_te = -1
    best_result = None
    beating_logit = []

    max_size = min(MAX_SUBSET_SIZE, len(all_features))
    for size in range(1, max_size + 1):
        for subset_tuple in combinations(all_features, size):
            subset = list(subset_tuple)
            cob_te, cob_tr, tree, logit_acc, feats = evaluate_subset_full(
                subset, train_clean, test_clean, col_t2
            )
            if cob_te is None:
                continue
            if cob_te > best_cob_te:
                best_cob_te = cob_te
                best_result = (feats, cob_te, cob_tr, tree, logit_acc)
            if baseline_logit is not None and cob_te > baseline_logit:
                beating_logit.append({
                    "features": feats,
                    "n": len(feats),
                    "arvore_teste": cob_te,
                    "logit": logit_acc,
                    "delta": round(cob_te - baseline_logit, 2)
                })

    beating_logit.sort(key=lambda x: x["delta"], reverse=True)
    return best_result, beating_logit[:10], baseline_logit

def residual_e_distribuicao(train_clean, test_clean, features, col_t2, uf):
    """Calcula residual e retorna perfil alto + bins de distribuição."""
    try:
        X_tr = train_clean[features]
        y_tr = train_clean[col_t2]
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        lin = LinearRegression()
        lin.fit(X_tr_s, y_tr)

        if len(test_clean) == 0:
            return None, None
        X_te = test_clean[features]
        y_te = test_clean[col_t2]
        pred = lin.predict(scaler.transform(X_te))
        residual = y_te - pred

        detail = test_clean.copy()
        detail["residual"] = residual.values
        detail["abs_residual"] = detail["residual"].abs()

        # Distribuição por faixas de 1 p.p.
        bins = np.arange(-20, 21, 1)
        counts, edges = np.histogram(detail["residual"].dropna(), bins=bins)
        dist = pd.DataFrame({
            "faixa_ini": edges[:-1],
            "faixa_fim": edges[1:],
            "n": counts,
        })
        dist = dist[dist["n"] > 0]
        dist.to_csv(RESULTS / f"residual_dist_{uf}.csv", index=False)

        # Perfil alto
        alto = detail[detail["abs_residual"] > RESIDUAL_ALTO]
        perfil = None
        if len(alto) >= 5:
            perfil = {
                "n_alto": len(alto),
                "pct_alto": round(100 * len(alto) / len(detail), 2),
                "res_medio": round(alto["residual"].mean(), 2),
            }

        return perfil, dist
    except Exception as e:
        print(f"  residual {uf}: {e}")
        return None, None

def main():
    join_files = sorted(PROCESSED.glob("join_*.parquet"))
    print(f"Joins pré-processados encontrados: {len(join_files)}")
    if not join_files:
        print(f"Nenhum join em {PROCESSED}. Rode preprocess_bu.py primeiro.")
        (RESULTS / "status.txt").write_text("Nenhum join pré-processado encontrado\n", encoding="utf-8")
        return

    summary = []
    perfil_rows = []
    all_dist = []
    for jf in join_files:
        uf = jf.stem.replace("join_", "")
        print(f"\nProcessando {uf}...")
        try:
            merged = pd.read_parquet(jf)
        except Exception as e:
            print(f"  Erro ao ler {jf.name}: {e}")
            continue
        print(f"  {len(merged)} seções")
        if len(merged) == 0:
            continue

        col_t1, col_t2 = find_best_candidate_match(merged)
        if not col_t1:
            print(f"  {uf}: nenhum match de candidato")
            continue
        print(f"  alvo = {col_t2}")

        train, test = split_mult3(merged)
        if train is None or len(train) < 30:
            print(f"  {uf}: treino insuficiente")
            continue

        all_features = get_features(merged)
        train_clean = train.dropna(subset=[col_t2] + all_features)
        test_clean = test.dropna(subset=[col_t2] + all_features)
        if len(train_clean) < 30:
            print(f"  {uf}: poucas seções válidas no treino após dropna ({len(train_clean)})")
            continue

        best_result, beating, baseline_logit = explore_combinations(
            all_features, train_clean, test_clean, col_t2
        )
        if not best_result:
            continue
        best_feats, cob_te, cob_tr, best_tree, logit_best = best_result

        perfil, dist = residual_e_distribuicao(train_clean, test_clean, all_features, col_t2, uf)
        if perfil:
            perfil_rows.append({"uf": uf, **perfil})
        if dist is not None and len(dist) > 0:
            dist["uf"] = uf
            all_dist.append(dist)

        rules = ""
        if best_tree is not None:
            rules = export_text(best_tree, feature_names=list(best_feats))[:3000]
        with open(RESULTS / f"regras_mult3_{uf}.txt", "w", encoding="utf-8") as f:
            f.write(f"Melhor árvore (combinatorial) — {uf}\n")
            f.write(f"Árvore teste: {cob_te}% | Logit: {baseline_logit}%\n")
            f.write("="*50 + "\n")
            f.write(rules + "\n")

        summary.append({
            "uf": uf,
            "n_treino": len(train_clean),
            "n_teste": len(test_clean),
            "arvore_teste": cob_te,
            "logit_baseline": baseline_logit,
        })

    status_lines = [f"UFs processadas: {len(summary)}"]
    if summary:
        with open(RESULTS / "resumo_mult3.txt", "w", encoding="utf-8") as f:
            f.write("Resumo mult3\n")
            f.write(f"{'UF':<6} {'Treino':>8} {'Teste':>8} {'ArvTe%':>8} {'Logit%':>8}\n")
            for m in summary:
                f.write(f"{m['uf']:<6} {m['n_treino']:>8} {m['n_teste']:>8} {str(m['arvore_teste']):>8} {str(m['logit_baseline']):>8}\n")
        status_lines.append("resumo_mult3.txt escrito")

    if perfil_rows:
        with open(RESULTS / "perfil_residual_alto.txt", "w", encoding="utf-8") as f:
            f.write(f"Seções com |residual| > {RESIDUAL_ALTO} p.p.\n")
            f.write("="*40 + "\n")
            f.write(f"{'UF':<6} {'NAlto':>7} {'%Alto':>7} {'ResMed':>8}\n")
            for r in perfil_rows:
                f.write(f"{r['uf']:<6} {r['n_alto']:>7} {r['pct_alto']:>7} {str(r.get('res_medio')):>8}\n")
        status_lines.append("perfil_residual_alto.txt escrito")

    if all_dist:
        full_dist = pd.concat(all_dist, ignore_index=True)
        # agrega nacional
        nacional = full_dist.groupby(["faixa_ini", "faixa_fim"])["n"].sum().reset_index()
        nacional.to_csv(RESULTS / "residual_dist_nacional.csv", index=False)
        status_lines.append("residual_dist_*.csv + residual_dist_nacional.csv escritos")

    (RESULTS / "status.txt").write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    print(f"\nResultados em {RESULTS.resolve()}")
    print("\n".join(status_lines))

if __name__ == "__main__":
    main()
