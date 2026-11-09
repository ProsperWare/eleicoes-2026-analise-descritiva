#!/usr/bin/env python3
"""
Exploração combinatorial + Random Forest no split mult3.
Compara árvore simples, logit e Random Forest (não-linear) na capacidade
de explicar o resultado do T2 a partir do T1 nos dados oficiais do TSE.
"""

import zipfile
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from itertools import combinations

RAW = Path("/home/alexa/eleicoes-data/raw")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)

SECTION_KEYS = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
TOLERANCE_PP = 5.0
MAX_DEPTH = 4
BINARY_THRESHOLD = 50.0
MIN_SAMPLES_LEAF = 15
MAX_SUBSET_SIZE = 5
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 8

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

def extract_cargo_pcts(df: pd.DataFrame, keys: list, cargo_code: str, top_n: int = 3, prefix: str = "") -> pd.DataFrame:
    if "QT_VOTOS" not in df.columns or "CD_CARGO" not in df.columns:
        return pd.DataFrame()
    work = df[df["CD_CARGO"] == cargo_code].copy()
    if work.empty:
        return pd.DataFrame()
    work["QT_VOTOS"] = to_numeric_safe(work["QT_VOTOS"])
    totals = work.groupby(keys)["QT_VOTOS"].sum().reset_index().rename(columns={"QT_VOTOS": "TOTAL_VOTOS"})
    cand_totals = work.groupby(["NR_VOTAVEL", "NM_VOTAVEL"])["QT_VOTOS"].sum().reset_index()
    top_cands = cand_totals.nlargest(top_n, "QT_VOTOS")["NR_VOTAVEL"].tolist()
    result = totals[keys].copy()
    for nr in top_cands:
        name = cand_totals.loc[cand_totals["NR_VOTAVEL"] == nr, "NM_VOTAVEL"].iloc[0]
        safe = "".join(c if c.isalnum() else "_" for c in str(name)[:30])
        col = f"PCT_{prefix}{safe}"
        cand_votes = work[work["NR_VOTAVEL"] == nr].groupby(keys)["QT_VOTOS"].sum().reset_index()
        cand_votes = cand_votes.merge(totals, on=keys, how="left")
        cand_votes[col] = (cand_votes["QT_VOTOS"] / cand_votes["TOTAL_VOTOS"].replace(0, pd.NA)) * 100
        result = result.merge(cand_votes[keys + [col]], on=keys, how="left")
    if cargo_code == "1":
        dir_votes = work[work["NR_VOTAVEL"].isin(DIREITA_PRES)].groupby(keys)["QT_VOTOS"].sum().reset_index()
        esq_votes = work[work["NR_VOTAVEL"].isin(ESQUERDA_PRES)].groupby(keys)["QT_VOTOS"].sum().reset_index()
        dir_votes = dir_votes.merge(totals, on=keys, how="left")
        esq_votes = esq_votes.merge(totals, on=keys, how="left")
        dir_votes["PCT_DIR_PRES"] = (dir_votes["QT_VOTOS"] / dir_votes["TOTAL_VOTOS"].replace(0, pd.NA)) * 100
        esq_votes["PCT_ESQ_PRES"] = (esq_votes["QT_VOTOS"] / esq_votes["TOTAL_VOTOS"].replace(0, pd.NA)) * 100
        result = result.merge(dir_votes[keys + ["PCT_DIR_PRES"]], on=keys, how="left")
        result = result.merge(esq_votes[keys + ["PCT_ESQ_PRES"]], on=keys, how="left")
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
    if is_t2:
        pcts = extract_cargo_pcts(df, keys, "1", top_n=2, prefix="T2_")
        if not pcts.empty:
            agg = agg.merge(pcts, on=keys, how="left")
    else:
        for cargo, prefix, top in [("1", "PRES_", 3), ("3", "GOV_", 2), ("5", "SEN_", 2), ("6", "DEP_", 2)]:
            pcts = extract_cargo_pcts(df, keys, cargo, top_n=top, prefix=prefix)
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

def main():
    zips = sorted(RAW.glob("bweb_*.zip"))
    print(f"Zips encontrados: {len(zips)} em {RAW}")
    if not zips:
        print(f"Nenhum zip em {RAW}")
        (RESULTS / "status.txt").write_text(f"Nenhum zip encontrado em {RAW}\n", encoding="utf-8")
        return
    section_dfs = {}
    for zp in zips:
        print(f"Processando {zp.name}...")
        try:
            raw = load_bu_zip(zp)
            label = zp.stem
            sec = aggregate_section(raw, label)
            section_dfs[label] = sec
            print(f"  -> {len(raw)} linhas → {len(sec)} seções | cols PCT: {sum(1 for c in sec.columns if c.startswith('PCT_'))}")
        except Exception as e:
            print(f"  ERRO: {e}")
    t1_keys = [k for k in section_dfs if "1t" in k]
    t2_keys = [k for k in section_dfs if "2t" in k]
    print(f"T1: {len(t1_keys)} | T2: {len(t2_keys)}")
    summary = []
    for t1k in t1_keys:
        uf = t1k.split("_")[2] if "_" in t1k else "?"
        t2k = next((k for k in t2_keys if uf in k), None)
        if not t2k:
            print(f"  {uf}: sem T2 correspondente")
            continue
        df1 = section_dfs[t1k]
        df2 = section_dfs[t2k]
        keys = [c for c in SECTION_KEYS if c in df1.columns and c in df2.columns]
        if not keys:
            print(f"  {uf}: chaves de seção ausentes")
            continue
        merged = df1.merge(df2, on=keys, how="inner", suffixes=("_t1", "_t2"))
        print(f"Join {uf}: {len(merged)} seções")
        if len(merged) == 0:
            continue
        if "QT_COMPARECIMENTO_t1" in merged.columns and "QT_COMPARECIMENTO_t2" in merged.columns:
            merged["diff_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] - merged["QT_COMPARECIMENTO_t1"]
            merged["razao_comparecimento"] = merged["QT_COMPARECIMENTO_t2"] / merged["QT_COMPARECIMENTO_t1"].replace(0, pd.NA)
        col_t1, col_t2 = find_best_candidate_match(merged)
        if not col_t1:
            print(f"  {uf}: nenhum match de candidato")
            continue
        print(f"  {uf}: alvo = {col_t2}")
        train, test = split_mult3(merged)
        if train is None or len(train) < 30:
            print(f"  {uf}: treino insuficiente ({0 if train is None else len(train)})")
            continue
        all_features = get_features(merged)
        train_clean = train.dropna(subset=[col_t2] + all_features)
        test_clean = test.dropna(subset=[col_t2] + all_features)
        if len(train_clean) < 30:
            print(f"  {uf}: poucas seções válidas no treino após dropna ({len(train_clean)})")
            continue
        print(f"  {uf}: {len(all_features)} features | explorando combinações até tamanho {MAX_SUBSET_SIZE}")

        best_result, beating, baseline_logit = explore_combinations(
            all_features, train_clean, test_clean, col_t2
        )
        if not best_result:
            print(f"  {uf}: nenhuma combinação válida")
            continue
        best_feats, cob_te, cob_tr, best_tree, logit_best = best_result

        # Random Forest com todas as features (baseline não-linear)
        rf_cob_te = None
        try:
            X_tr_all = train_clean[all_features]
            y_tr_all = train_clean[col_t2]
            X_te_all = test_clean[all_features] if len(test_clean) > 0 else pd.DataFrame()
            y_te_all = test_clean[col_t2] if len(test_clean) > 0 else pd.Series(dtype=float)
            rf = RandomForestRegressor(
                n_estimators=RF_N_ESTIMATORS,
                max_depth=RF_MAX_DEPTH,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=42,
                n_jobs=-1
            )
            rf.fit(X_tr_all, y_tr_all)
            rf_cob_te, _ = evaluate_regression(rf, X_te_all, y_te_all) if len(X_te_all) > 0 else (None, None)
            print(f"  {uf}: Random Forest teste={rf_cob_te}%")
        except Exception as e:
            print(f"  {uf}: erro no RF: {e}")
            rf_cob_te = None

        print(f"  {uf}: melhor árvore={cob_te}% | logit={baseline_logit}% | RF={rf_cob_te}% | {len(beating)} subsets > logit")

        rules = ""
        if best_tree is not None:
            rules = export_text(best_tree, feature_names=list(best_feats))[:3000]
        with open(RESULTS / f"regras_mult3_{uf}.txt", "w", encoding="utf-8") as f:
            f.write(f"Melhor árvore (combinatorial) + comparação RF — {uf}\n")
            f.write(f"Features melhor árvore ({len(best_feats)}): {', '.join(best_feats)}\n")
            f.write(f"Árvore teste: {cob_te}% | treino: {cob_tr}%\n")
            f.write(f"Logit baseline: {baseline_logit}%\n")
            f.write(f"Random Forest (todas features): {rf_cob_te}%\n")
            f.write(f"Subsets árvore > logit: {len(beating)}\n")
            f.write("="*50 + "\n")
            f.write(rules + "\n")
            if beating:
                f.write("\nTop subsets que superam o logit:\n")
                for b in beating[:5]:
                    f.write(f"  {b['n']} feats | árvore={b['arvore_teste']}% | delta={b['delta']}% | {', '.join(b['features'][:3])}...\n")

        summary.append({
            "uf": uf,
            "n_treino": len(train_clean),
            "n_teste": len(test_clean),
            "n_features_best": len(best_feats),
            "arvore_teste": cob_te,
            "logit_baseline": baseline_logit,
            "rf_teste": rf_cob_te,
            "n_beating": len(beating),
            "best_delta": beating[0]["delta"] if beating else None,
        })

    status_lines = [f"UFs processadas: {len(summary)}"]
    if summary:
        with open(RESULTS / "resumo_mult3.txt", "w", encoding="utf-8") as f:
            f.write("Resumo: combinatorial + Random Forest no split mult3\n")
            f.write("Compara árvore simples, logit e RF (não-linear)\n")
            f.write("Caminho dos dados: /home/alexa/eleicoes-data/raw\n")
            f.write("="*100 + "\n")
            f.write(f"{'UF':<6} {'Treino':>8} {'Teste':>8} {'NFeat':>6} {'ArvTe%':>8} {'Logit%':>8} {'RF%':>8} {'NBeating':>9} {'BestDelta':>10}\n")
            f.write("-"*100 + "\n")
            for m in summary:
                f.write(f"{m['uf']:<6} {m['n_treino']:>8} {m['n_teste']:>8} {m['n_features_best']:>6} {str(m['arvore_teste']):>8} {str(m['logit_baseline']):>8} {str(m['rf_teste']):>8} {m['n_beating']:>9} {str(m['best_delta']):>10}\n")
        status_lines.append("resumo_mult3.txt escrito (com RF)")
    else:
        status_lines.append("Nenhuma UF gerou resumo")
    (RESULTS / "status.txt").write_text("\n".join(status_lines) + "\n", encoding="utf-8")
    print(f"\nResultados em {RESULTS.resolve()}")
    print("\n".join(status_lines))

if __name__ == "__main__":
    main()
