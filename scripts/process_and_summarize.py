#!/usr/bin/env python3
"""
Residual do linear (T1->T2) + distribuição + residual por zona (todas UFs).
Marca zonas com desvio em relação à normal esperada.
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

        bins = np.arange(-15, 15.1, 0.1)
        counts, edges = np.histogram(residual.dropna(), bins=bins)
        dist = pd.DataFrame({
            "faixa_ini": edges[:-1].round(1),
            "faixa_fim": edges[1:].round(1),
            "n": counts,
        })
        dist = dist[dist["n"] > 0]
        dist.to_csv(RESULTS / f"residual_dist_{uf}.csv", index=False)

        info = {
            "n": len(residual),
            "res_medio": round(float(residual.mean()), 3),
            "res_std": round(float(residual.std()), 3),
            "pct_dentro_5": round(100 * (residual.abs() <= 5).mean(), 2),
        }

        res_df = None
        if "NR_ZONA" in test_clean.columns:
            res_df = test_clean[["NR_ZONA"]].copy()
            res_df["residual"] = residual.values
            res_df = res_df.dropna(subset=["residual"])
            res_df["uf"] = uf
        return info, dist, res_df
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
    all_dist = []
    all_zona = []

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

        info, dist, res_df = residual_distribuicao(train_clean, test_clean, features, col_t2, uf)
        if info and dist is not None:
            rows.append({"uf": uf, **info})
            dist = dist.copy()
            dist["uf"] = uf
            all_dist.append(dist)
            print(f"  n={info['n']} | dentro ±5pp={info['pct_dentro_5']}%")

            if res_df is not None and len(res_df) > 0:
                zona_agg = res_df.groupby(["uf", "NR_ZONA"])["residual"].agg(
                    n="count",
                    res_medio="mean",
                    res_std="std",
                    pct_dentro_5=lambda x: 100 * (x.abs() <= 5).mean(),
                ).reset_index()
                zona_agg["res_medio"] = zona_agg["res_medio"].round(3)
                zona_agg["res_std"] = zona_agg["res_std"].round(3)
                zona_agg["pct_dentro_5"] = zona_agg["pct_dentro_5"].round(2)
                all_zona.append(zona_agg)

    if rows:
        with open(RESULTS / "residual_resumo.txt", "w", encoding="utf-8") as f:
            f.write("Residual do linear (teste)\n")
            f.write(f"{'UF':<6} {'N':>8} {'ResMed':>8} {'ResStd':>8} {'%<=5pp':>8}\n")
            f.write("-"*40 + "\n")
            for r in rows:
                f.write(f"{r['uf']:<6} {r['n']:>8} {r['res_medio']:>8} {r['res_std']:>8} {r['pct_dentro_5']:>8}\n")

    if all_dist:
        full = pd.concat(all_dist, ignore_index=True)
        nacional = full.groupby(["faixa_ini", "faixa_fim"])["n"].sum().reset_index()
        nacional["densidade"] = nacional["n"] / nacional["n"].sum()
        nacional.to_csv(RESULTS / "residual_dist_nacional.csv", index=False)

        uf_to_regiao = {u: reg for reg, ufs in REGIOES.items() for u in ufs}
        full["regiao"] = full["uf"].map(uf_to_regiao)
        for reg in REGIOES.keys():
            sub = full[full["regiao"] == reg]
            if len(sub) == 0:
                continue
            agg = sub.groupby(["faixa_ini", "faixa_fim"])["n"].sum().reset_index()
            agg["densidade"] = agg["n"] / agg["n"].sum()
            agg.to_csv(RESULTS / f"residual_dist_regiao_{reg.replace(' ', '_')}.csv", index=False)

    if all_zona:
        zonas = pd.concat(all_zona, ignore_index=True)
        zonas.to_csv(RESULTS / "residual_zona_todas.csv", index=False)

        # Flag de desvio: |res_medio| > 1.5 ou pct_dentro_5 < 95 ou res_std > 3
        desvio = zonas[
            (zonas["res_medio"].abs() > 1.5) |
            (zonas["pct_dentro_5"] < 95) |
            (zonas["res_std"] > 3.0)
        ].copy()
        desvio = desvio.sort_values("res_medio", key=abs, ascending=False)
        desvio.to_csv(RESULTS / "residual_zona_desvio.csv", index=False)
        print(f"Zonas com desvio: {len(desvio)} de {len(zonas)}")

    status = f"UFs processadas: {len(rows)}\nZonas totais: {len(zonas) if all_zona else 0}\n"
    (RESULTS / "status.txt").write_text(status, encoding="utf-8")
    print(status)

if __name__ == "__main__":
    main()
