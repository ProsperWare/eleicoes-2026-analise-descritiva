#!/usr/bin/env python3
"""
Processa BUs, gera baseline, árvore, linear, logit, clusters e
módulo preditivo sequencial (múltiplos de 3 → pares restantes → ímpares restantes).
"""

import zipfile
from pathlib import Path
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

RAW = Path("data/raw")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)

SECTION_KEYS = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
TOLERANCE_PP = 5.0
MAX_DEPTH = 4
BINARY_THRESHOLD = 50.0
N_CLUSTERS = 4

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

def modulo_preditivo_sequencial(merged: pd.DataFrame, col_t2: str) -> dict:
    if "NR_SECAO" not in merged.columns:
        return {"erro": "NR_SECAO não encontrado"}
    features = get_features(merged)
    work = merged[features + [col_t2, "NR_SECAO"]].dropna().copy()
    if len(work) < 100:
        return {"erro": "Dados insuficientes"}
    work["NR_SECAO_num"] = to_numeric_safe(work["NR_SECAO"])
    train = work[work["NR_SECAO_num"] % 3 == 0]
    remaining = work[work["NR_SECAO_num"] % 3 != 0]
    test_even = remaining[remaining["NR_SECAO_num"] % 2 == 0]
    test_odd = remaining[remaining["NR_SECAO_num"] % 2 != 0]
    if len(train) < 30:
        return {"erro": "Treino insuficiente"}
    X_train = train[features]
    y_train = train[col_t2]
    tree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=42, min_samples_leaf=15)
    tree.fit(X_train, y_train)
    def evaluate(subset, name):
        if len(subset) < 10:
            return {"nome": name, "n": len(subset), "cobertura": None, "erro_medio": None}
        X = subset[features]
        y = subset[col_t2]
        pred = tree.predict(X)
        erro = np.abs(pred - y)
        within = (erro <= TOLERANCE_PP).sum()
        cob = 100 * within / len(subset)
        return {"nome": name, "n": len(subset), "cobertura": round(cob, 2), "erro_medio": round(float(erro.mean()), 2)}
    results = [
        evaluate(train, "treino_multiplos_3"),
        evaluate(test_even, "teste_pares_restantes"),
        evaluate(test_odd, "teste_impares_restantes"),
    ]
    rules = export_text(tree, feature_names=list(features))[:2000]
    return {"resultados": results, "regras": rules, "n_treino": len(train), "n_teste_pares": len(test_even), "n_teste_impares": len(test_odd)}

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
            continue
        # Módulo preditivo sequencial
        pred_seq = modulo_preditivo_sequencial(merged, col_t2)
        if "resultados" in pred_seq:
            with open(RESULTS / f"preditivo_sequencial_{uf}.txt", "w", encoding="utf-8") as f:
                f.write(f"Módulo preditivo sequencial — {uf}\n")
                f.write("="*50 + "\n")
                f.write(f"Treino (múltiplos de 3): {pred_seq['n_treino']} seções\n")
                f.write(f"Teste pares: {pred_seq['n_teste_pares']} | Ímpares: {pred_seq['n_teste_impares']}\n\n")
                for r in pred_seq["resultados"]:
                    f.write(f"{r['nome']}: cobertura {r['cobertura']}% | erro {r['erro_medio']} p.p. (n={r['n']})\n")
                f.write("\nRegras da árvore (treinada nos múltiplos de 3):\n")
                f.write(pred_seq["regras"] + "\n")
            # Extrai cobertura dos testes para o resumo
            cob_pares = next((r["cobertura"] for r in pred_seq["resultados"] if r["nome"] == "teste_pares_restantes"), None)
            cob_impares = next((r["cobertura"] for r in pred_seq["resultados"] if r["nome"] == "teste_impares_restantes"), None)
        else:
            cob_pares = cob_impares = None
        model_summary.append({
            "uf": uf,
            "secoes": len(merged),
            "cob_pares": cob_pares,
            "cob_impares": cob_impares,
        })
    if model_summary:
        with open(RESULTS / "resumo_preditivo_sequencial.txt", "w", encoding="utf-8") as f:
            f.write("Resumo módulo preditivo sequencial\n")
            f.write("="*50 + "\n")
            f.write(f"{'UF':<6} {'Seções':>8} {'Pares%':>10} {'Ímpares%':>12}\n")
            f.write("-"*40 + "\n")
            for m in model_summary:
                f.write(f"{m['uf']:<6} {m['secoes']:>8} {str(m['cob_pares']):>10} {str(m['cob_impares']):>12}\n")
    print(f"\nResultados em {RESULTS.resolve()}")

if __name__ == "__main__":
    main()
