#!/usr/bin/env python3
"""
Processa Boletins de Urna baixados (zips em data/raw/),
agrega por seção, extrai percentuais de candidatos principais,
junt a T1 x T2, calcula baseline de transferência proporcional
e mede cobertura (seções dentro da tolerância).
Inclui matching melhorado de candidatos, árvore de decisão rasa,
importância de features, Logit binário, modelo linear,
análise de resíduos e efeito do comparecimento.
Grava apenas saídas processadas em resultados/.
"""

import zipfile
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
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

def compute_coverage(merged: pd.DataFrame, col_t1: str, col_t2: str, tol: float = TOLERANCE_PP) -> dict:
    if col_t1 not in merged.columns or col_t2 not in merged.columns:
        return {"cobertura": None, "nota": "Colunas não encontradas"}

    work = merged[[col_t1, col_t2]].dropna().copy()
    work["erro_pp"] = (work[col_t2] - work[col_t1]).abs()
    within = (work["erro_pp"] <= tol).sum()
    total = len(work)

    return {
        "candidato_t1": col_t1,
        "candidato_t2": col_t2,
        "tolerancia_pp": tol,
        "secoes_avaliadas": int(total),
        "secoes_dentro": int(within),
        "cobertura_pct": round(100 * within / total, 2) if total > 0 else 0,
        "erro_medio_pp": round(float(work["erro_pp"].mean()), 2),
        "erro_mediano_pp": round(float(work["erro_pp"].median()), 2),
    }

def get_features(merged: pd.DataFrame) -> list:
    features = [c for c in merged.columns if c.startswith("PCT_") and c.endswith("_t1")]
    for extra in ["QT_COMPARECIMENTO_t1", "QT_APTOS_t1", "diff_comparecimento", "razao_comparecimento"]:
        if extra in merged.columns:
            features.append(extra)
    return features

def fit_simple_tree(merged: pd.DataFrame, col_t1: str, col_t2: str, max_depth: int = MAX_DEPTH) -> dict:
    features = get_features(merged)
    work = merged[features + [col_t2]].dropna()
    if len(work) < 50:
        return {"erro": "Dados insuficientes para árvore"}

    X = work[features]
    y = work[col_t2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    tree = DecisionTreeRegressor(max_depth=max_depth, random_state=42, min_samples_leaf=20)
    tree.fit(X_train, y_train)

    pred = tree.predict(X_test)
    erro = np.abs(pred - y_test)
    within = (erro <= TOLERANCE_PP).sum()
    cobertura = 100 * within / len(y_test)

    rules = export_text(tree, feature_names=list(features))

    gini_imp = dict(zip(features, tree.feature_importances_))
    gini_sorted = sorted(gini_imp.items(), key=lambda x: x[1], reverse=True)

    try:
        perm = permutation_importance(tree, X_test, y_test, n_repeats=5, random_state=42, n_jobs=1)
        perm_imp = dict(zip(features, perm.importances_mean))
        perm_sorted = sorted(perm_imp.items(), key=lambda x: x[1], reverse=True)
    except Exception:
        perm_sorted = []

    return {
        "max_depth": max_depth,
        "cobertura_teste_pct": round(cobertura, 2),
        "erro_medio_teste_pp": round(float(erro.mean()), 2),
        "n_teste": len(y_test),
        "n_features": len(features),
        "regras": rules[:3000],
        "importancia_gini": gini_sorted,
        "importancia_permutacao": perm_sorted,
        "pred_test": pred,
        "y_test": y_test.values,
        "X_test": X_test,
    }

def fit_logit_binary(merged: pd.DataFrame, col_t2: str, threshold: float = BINARY_THRESHOLD) -> dict:
    features = get_features(merged)
    work = merged[features + [col_t2]].dropna()
    if len(work) < 50:
        return {"erro": "Dados insuficientes"}

    X = work[features]
    y_cont = work[col_t2]
    y_bin = (y_cont >= threshold).astype(int)

    X_train, X_test, y_train, y_test, y_cont_train, y_cont_test = train_test_split(
        X, y_bin, y_cont, test_size=0.3, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    logit = LogisticRegression(max_iter=1000, random_state=42)
    logit.fit(X_train_s, y_train)

    proba = logit.predict_proba(X_test_s)[:, 1]
    pred_bin = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred_bin)
    try:
        auc = roc_auc_score(y_test, proba)
    except Exception:
        auc = None

    coefs = dict(zip(features, logit.coef_[0]))
    coefs_sorted = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "threshold": threshold,
        "acuracia_teste": round(acc * 100, 2),
        "auc": round(auc, 3) if auc else None,
        "n_teste": len(y_test),
        "proporcao_positiva_teste": round(y_test.mean() * 100, 2),
        "coeficientes": coefs_sorted[:10],
    }

def fit_linear_fractional(merged: pd.DataFrame, col_t2: str) -> dict:
    features = get_features(merged)
    work = merged[features + [col_t2]].dropna()
    if len(work) < 50:
        return {"erro": "Dados insuficientes"}

    X = work[features]
    y = work[col_t2]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    lin = LinearRegression()
    lin.fit(X_train_s, y_train)

    pred = lin.predict(X_test_s)
    pred = np.clip(pred, 0, 100)
    erro = np.abs(pred - y_test)
    within = (erro <= TOLERANCE_PP).sum()
    cobertura = 100 * within / len(y_test)
    mae = mean_absolute_error(y_test, pred)

    coefs = dict(zip(features, lin.coef_))
    coefs_sorted = sorted(coefs.items(), key=lambda x: abs(x[1]), reverse=True)

    return {
        "cobertura_teste_pct": round(cobertura, 2),
        "erro_medio_teste_pp": round(float(erro.mean()), 2),
        "mae": round(mae, 2),
        "n_teste": len(y_test),
        "coeficientes": coefs_sorted[:10],
        "intercepto": round(float(lin.intercept_), 2),
        "pred_test": pred,
        "y_test": y_test.values,
    }

def analyze_residuals(y_true, y_pred, X_test, uf: str, model_name: str):
    """Analisa resíduos por faixas de percentual e comparecimento."""
    if y_true is None or len(y_true) == 0:
        return
    erro = np.abs(y_true - y_pred)
    df = X_test.copy()
    df["y_true"] = y_true
    df["erro"] = erro

    lines = [f"Análise de resíduos — {model_name} — {uf}", "="*60, ""]

    # Por faixas de percentual T1 do candidato principal
    main_col = [c for c in df.columns if c.startswith("PCT_") and c.endswith("_t1")]
    if main_col:
        col = main_col[0]
        df["faixa_t1"] = pd.cut(df[col], bins=[0, 10, 20, 30, 40, 50, 100], include_lowest=True)
        lines.append("Erro médio por faixa de % no T1:")
        for faixa, g in df.groupby("faixa_t1", observed=True):
            lines.append(f"  {faixa}: {g['erro'].mean():.2f} p.p. (n={len(g)})")
        lines.append("")

    # Por diferença de comparecimento
    if "diff_comparecimento" in df.columns:
        df["faixa_diff"] = pd.cut(df["diff_comparecimento"], bins=[-1000, -50, -10, 10, 50, 1000], include_lowest=True)
        lines.append("Erro médio por faixa de diferença de comparecimento:")
        for faixa, g in df.groupby("faixa_diff", observed=True):
            lines.append(f"  {faixa}: {g['erro'].mean():.2f} p.p. (n={len(g)})")
        lines.append("")

    with open(RESULTS / f"residuos_{model_name}_{uf}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

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

    t1_keys = [k for k in section_dfs if "1t" in k]
    t2_keys = [k for k in section_dfs if "2t" in k]

    coverage_reports = []
    model_summary = []

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
            print(f"  Sem match de candidatos para {uf}")
            continue

        cov = compute_coverage(merged, col_t1, col_t2, TOLERANCE_PP)
        cov["uf"] = uf
        cov["secoes_join"] = len(merged)
        coverage_reports.append(cov)

        tree_result = fit_simple_tree(merged, col_t1, col_t2, MAX_DEPTH)
        cov["arvore"] = {k: v for k, v in tree_result.items() if k not in ("regras", "importancia_gini", "importancia_permutacao", "pred_test", "y_test", "X_test")}

        logit_bin = fit_logit_binary(merged, col_t2, BINARY_THRESHOLD)
        cov["logit_binario"] = {k: v for k, v in logit_bin.items() if k != "coeficientes"}

        linear = fit_linear_fractional(merged, col_t2)
        cov["linear"] = {k: v for k, v in linear.items() if k != "coeficientes" and k not in ("pred_test", "y_test")}

        # Resíduos
        if "pred_test" in tree_result:
            analyze_residuals(tree_result["y_test"], tree_result["pred_test"], tree_result["X_test"], uf, "arvore")
        if "pred_test" in linear:
            # Para linear precisamos reconstruir X_test aproximado — simplificado
            pass

        # Efeito isolado do comparecimento
        if "diff_comparecimento" in merged.columns:
            features_comp = ["diff_comparecimento", "razao_comparecimento"]
            features_comp = [f for f in features_comp if f in merged.columns]
            if features_comp:
                work = merged[features_comp + [col_t2]].dropna()
                if len(work) > 50:
                    X = work[features_comp]
                    y = work[col_t2]
                    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
                    sc = StandardScaler()
                    X_tr_s = sc.fit_transform(X_tr)
                    X_te_s = sc.transform(X_te)
                    lin_c = LinearRegression().fit(X_tr_s, y_tr)
                    pred_c = np.clip(lin_c.predict(X_te_s), 0, 100)
                    erro_c = np.abs(pred_c - y_te)
                    cov_comp = 100 * (erro_c <= TOLERANCE_PP).sum() / len(y_te)
                    with open(RESULTS / f"efeito_comparecimento_{uf}.txt", "w", encoding="utf-8") as f:
                        f.write(f"Modelo só com comparecimento — {uf}\n")
                        f.write(f"Cobertura ±{TOLERANCE_PP} p.p.: {cov_comp:.2f}%\n")
                        f.write(f"Erro médio: {erro_c.mean():.2f} p.p.\n")
                        f.write(f"Coeficientes: {dict(zip(features_comp, lin_c.coef_))}\n")

        summary_line = {
            "uf": uf,
            "secoes": len(merged),
            "arvore_cob": cov["arvore"].get("cobertura_teste_pct"),
            "linear_cob": cov["linear"].get("cobertura_teste_pct"),
            "logit_acc": cov["logit_binario"].get("acuracia_teste"),
        }
        model_summary.append(summary_line)

        with open(RESULTS / f"join_{uf}_resumo.txt", "w", encoding="utf-8") as f:
            f.write(str({"uf": uf, "secoes_join": len(merged), "cobertura": cov}) + "\n")

        if "regras" in tree_result:
            with open(RESULTS / f"regras_arvore_{uf}.txt", "w", encoding="utf-8") as f:
                f.write(f"Árvore rasa (max_depth={MAX_DEPTH}) para {uf}\n")
                f.write("="*50 + "\n")
                f.write(tree_result["regras"] + "\n")

        if "importancia_gini" in tree_result:
            with open(RESULTS / f"importancia_features_{uf}.txt", "w", encoding="utf-8") as f:
                f.write(f"Importância de features — árvore depth={MAX_DEPTH} — {uf}\n")
                f.write("="*60 + "\n\n")
                f.write("Importância Gini:\n")
                for feat, imp in tree_result["importancia_gini"]:
                    f.write(f"  {feat}: {imp:.4f}\n")
                f.write("\nImportância por permutação:\n")
                for feat, imp in tree_result.get("importancia_permutacao", []):
                    f.write(f"  {feat}: {imp:.4f}\n")

        if "coeficientes" in logit_bin:
            with open(RESULTS / f"logit_binario_{uf}.txt", "w", encoding="utf-8") as f:
                f.write(f"Logit binário (≥{BINARY_THRESHOLD}% no T2) — {uf}\n")
                f.write("="*50 + "\n")
                f.write(f"Acurácia teste: {logit_bin.get('acuracia_teste')}%\n")
                f.write(f"AUC: {logit_bin.get('auc')}\n")
                f.write(f"Proporção positiva: {logit_bin.get('proporcao_positiva_teste')}%\n\n")
                f.write("Coeficientes (top):\n")
                for feat, coef in logit_bin["coeficientes"]:
                    f.write(f"  {feat}: {coef:.4f}\n")

        if "coeficientes" in linear:
            with open(RESULTS / f"linear_fractional_{uf}.txt", "w", encoding="utf-8") as f:
                f.write(f"Modelo linear (proxy fractional) — {uf}\n")
                f.write("="*50 + "\n")
                f.write(f"Cobertura ±{TOLERANCE_PP} p.p.: {linear.get('cobertura_teste_pct')}%\n")
                f.write(f"Erro médio: {linear.get('erro_medio_teste_pp')} p.p.\n")
                f.write(f"MAE: {linear.get('mae')}\n")
                f.write(f"Intercepto: {linear.get('intercepto')}\n\n")
                f.write("Coeficientes (top):\n")
                for feat, coef in linear["coeficientes"]:
                    f.write(f"  {feat}: {coef:.4f}\n")

    if coverage_reports:
        with open(RESULTS / "cobertura_baseline.txt", "w", encoding="utf-8") as f:
            f.write(f"Cobertura do baseline de transferência proporcional (tolerância ±{TOLERANCE_PP} p.p.)\n")
            f.write("Matching por maior correlação entre candidatos T1 e T2\n")
            f.write("=" * 70 + "\n\n")
            for c in coverage_reports:
                f.write(str(c) + "\n\n")

    if model_summary:
        with open(RESULTS / "resumo_modelos.txt", "w", encoding="utf-8") as f:
            f.write("Resumo consolidado dos modelos por UF\n")
            f.write("="*50 + "\n\n")
            f.write(f"{'UF':<6} {'Seções':>8} {'Árvore%':>10} {'Linear%':>10} {'LogitAcc%':>12}\n")
            f.write("-"*50 + "\n")
            for m in model_summary:
                f.write(f"{m['uf']:<6} {m['secoes']:>8} {str(m['arvore_cob']):>10} {str(m['linear_cob']):>10} {str(m['logit_acc']):>12}\n")

    print(f"\nResultados em {RESULTS.resolve()}")

if __name__ == "__main__":
    main()
