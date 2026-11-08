from sklearn.cluster import KMeans

def fit_clusters(merged: pd.DataFrame, col_t2: str, n_clusters: int = 4) -> dict:
    """Clusteriza seções e ajusta árvore local por cluster."""
    features = get_features(merged)
    work = merged[features + [col_t2]].dropna()
    if len(work) < 100:
        return {"erro": "Dados insuficientes para clustering"}

    X = work[features]
    y = work[col_t2]

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_s)
    work = work.copy()
    work["cluster"] = clusters

    results = []
    total_within = 0
    total_test = 0

    for c in range(n_clusters):
        subset = work[work["cluster"] == c]
        if len(subset) < 30:
            continue

        Xc = subset[features]
        yc = subset[col_t2]
        X_tr, X_te, y_tr, y_te = train_test_split(Xc, yc, test_size=0.3, random_state=42)

        if len(X_tr) < 10:
            continue

        tree = DecisionTreeRegressor(max_depth=3, random_state=42, min_samples_leaf=10)
        tree.fit(X_tr, y_tr)
        pred = tree.predict(X_te)
        erro = np.abs(pred - y_te)
        within = (erro <= TOLERANCE_PP).sum()
        cob = 100 * within / len(y_te) if len(y_te) > 0 else 0

        total_within += within
        total_test += len(y_te)

        rules = export_text(tree, feature_names=list(features))[:1500]

        results.append({
            "cluster": c,
            "n": len(subset),
            "cobertura_teste": round(cob, 2),
            "erro_medio": round(float(erro.mean()), 2),
            "regras": rules,
        })

    cobertura_global = 100 * total_within / total_test if total_test > 0 else 0

    return {
        "n_clusters": n_clusters,
        "cobertura_global_teste": round(cobertura_global, 2),
        "clusters": results,
    }
