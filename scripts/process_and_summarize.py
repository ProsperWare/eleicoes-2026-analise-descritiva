def modulo_preditivo_sequencial(merged: pd.DataFrame, col_t2: str) -> dict:
    """
    Módulo preditivo sequencial:
    1. Treina nas seções múltiplas de 3
    2. Testa nas pares restantes
    3. Testa nas ímpares restantes
    """
    if "NR_SECAO" not in merged.columns:
        return {"erro": "NR_SECAO não encontrado"}

    features = get_features(merged)
    work = merged[features + [col_t2, "NR_SECAO"]].dropna().copy()
    if len(work) < 100:
        return {"erro": "Dados insuficientes"}

    work["NR_SECAO_num"] = to_numeric_safe(work["NR_SECAO"])

    # 1. Treino: múltiplos de 3
    train = work[work["NR_SECAO_num"] % 3 == 0]
    remaining = work[work["NR_SECAO_num"] % 3 != 0]

    # 2. Teste pares restantes
    test_even = remaining[remaining["NR_SECAO_num"] % 2 == 0]
    # 3. Teste ímpares restantes
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
        return {
            "nome": name,
            "n": len(subset),
            "cobertura": round(cob, 2),
            "erro_medio": round(float(erro.mean()), 2),
        }

    results = [
        evaluate(train, "treino_multiplos_3"),
        evaluate(test_even, "teste_pares_restantes"),
        evaluate(test_odd, "teste_impares_restantes"),
    ]

    rules = export_text(tree, feature_names=list(features))[:2000]

    return {
        "resultados": results,
        "regras": rules,
        "n_treino": len(train),
        "n_teste_pares": len(test_even),
        "n_teste_impares": len(test_odd),
    }
