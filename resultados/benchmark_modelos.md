# Benchmark de modelos vs logit

## Protocolo

O alvo é o mesmo de `process_and_summarize.py`: o percentual do segundo turno escolhido por `find_best_candidate_match` (a coluna `PCT_*_t2` com maior correlação com algum `PCT_*_t1`). A unidade é ponto percentual. O baseline `logit` é o modelo vigente em `residual_distribuicao`: `StandardScaler` + `LinearRegression`, ajustado sem peso amostral. RMSE, MAE e R² também são calculados com peso de eleitores.

O benchmark reestima o logit neste protocolo de três grupos. Os arquivos `residual_secao_*.parquet` já publicados foram gerados por `split_mult3`, que ajusta o linear nas seções com `NR_SECAO` múltiplo de 3; este benchmark não reutiliza esses resíduos.

### Divisão por NR_SECAO

Cada seção entra em um único grupo, pelo número inteiro de `NR_SECAO`:

| Grupo | Regra | Papel principal |
| --- | --- | --- |
| Grupo A | múltiplo de 3 | teste |
| Grupo B | múltiplo de 2 e não múltiplo de 3 | validação |
| Grupo C | o restante (nem múltiplo de 2 nem de 3) | treino |

Múltiplos de 6 são múltiplos de 3 e ficam no Grupo A. Seção sem número inteiro fica de fora.

No ajuste principal o treino é o Grupo C, a validação (tuning e escolha do modelo) é o Grupo B e o teste final é o Grupo A. O modelo treinado em C também é medido nos três grupos.

Rotação — cada grupo como teste uma vez. Os hiperparâmetros são reescolhidos só na validação daquela rotação; o teste não entra no ajuste:

| Teste | Validação | Treino |
| --- | --- | --- |
| Grupo A | Grupo B | Grupo C |
| Grupo B | Grupo C | Grupo A |
| Grupo C | Grupo A | Grupo B |

A validação cruzada agrupada por município (`GroupKFold`, 3 folds; por zona se não houver municípios suficientes) usa somente as seções dos Grupos B e C, no experimento base, com hiperparâmetros já fixados. Ela não escolhe o modelo.

### Experimentos

- `base`: as mesmas features de `get_features` (percentuais do primeiro turno e, quando existem, comparecimento, aptos, diferença e razão de comparecimento). Um ajuste por UF, como o logit atual.
- `extra`: as features base mais o logaritmo do tamanho da seção (aptos), a abstenção do primeiro turno e o target encoding do município (média suavizada do alvo, fora da amostra no treino, m=20). Um ajuste por UF.
- `extra_uf`: quando várias UFs compartilham o mesmo alvo, um modelo nacional na interseção das features base, com tamanho, abstenção, target encoding (chave UF|município) e dummies de UF.

### Modelos

- `logit`: regressão linear (baseline atual)
- `ridge`: regressão linear com L2
- `fractional_logit`: logit fracionário com L2 (IRLS, ligação logit)
- `hist_gradient_boosting`: `HistGradientBoostingRegressor`
- `random_forest`: `RandomForestRegressor` (40 árvores; acima de 60 mil linhas, cada árvore vê no máximo 40 mil)
- `logit_ols`: ridge no logit da proporção, invertido para pontos percentuais (aproximação beta/fracionária)

O tuning de árvore, quando o treino passa de 5 mil linhas, usa uma subamostra de 5 mil e refaz o ajuste vencedor no treino inteiro. A escolha do modelo minimiza o RMSE ponderado na validação do Grupo B. O teste A não entra nessa escolha.

### Métricas

RMSE, MAE e R² em pontos percentuais, nas versões simples e ponderada pelos eleitores (`QT_APTOS_t2`, senão `QT_APTOS_t1`, senão comparecimento). Também a porcentagem de seções com erro absoluto até 5 p.p., o log-loss e o Brier na escala de proporção. Nessas duas últimas a previsão é cortada em (1e-6, 1−1e-6); RMSE e MAE usam a previsão crua, como o residual do logit.

O resíduo gravado é observado − previsto no teste A, apenas para o modelo selecionado na validação, e só quando o arquivo cabe em um parquet enxuto.

Nenhum resultado numérico foi gerado nesta execução. O runner grava `benchmark_modelos_{ano}.csv` ao processar os joins.

