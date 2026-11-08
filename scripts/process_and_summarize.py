def fit_logit_binary(merged: pd.DataFrame, col_t2: str, threshold: float = BINARY_THRESHOLD) -> dict:
    features = get_features(merged)
    work = merged[features + [col_t2]].dropna()
    if len(work) < 50:
        return {"erro": "Dados insuficientes"}

    X = work[features]
    y_cont = work[col_t2]
    y_bin = (y_cont >= threshold).astype(int)

    # Se só existe uma classe, não dá para treinar logit
    if y_bin.nunique() < 2:
        return {
            "threshold": threshold,
            "erro": "Apenas uma classe no alvo binário",
            "acuracia_teste": None,
            "auc": None,
            "n_teste": len(work),
            "proporcao_positiva_teste": round(y_bin.mean() * 100, 2),
            "coeficientes": [],
        }

    X_train, X_test, y_train, y_test, y_cont_train, y_cont_test = train_test_split(
        X, y_bin, y_cont, test_size=0.3, random_state=42, stratify=y_bin
    )

    # Verifica novamente no treino
    if y_train.nunique() < 2:
        return {
            "threshold": threshold,
            "erro": "Apenas uma classe no treino",
            "acuracia_teste": None,
            "auc": None,
            "n_teste": len(y_test),
            "proporcao_positiva_teste": round(y_test.mean() * 100, 2),
            "coeficientes": [],
        }

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    try:
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
    except Exception as e:
        return {"erro": str(e), "acuracia_teste": None, "auc": None, "coeficientes": []}
