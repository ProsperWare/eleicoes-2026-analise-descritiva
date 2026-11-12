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

## Resultado

- 2018: selecionado **ridge** no experimento `base` (teste A, RMSE ponderado 2.424; logit base 2.424).
- 2022: selecionado **ridge** no experimento `base` (teste A, RMSE ponderado 1.821; logit base 1.821).
- 2026: selecionado **ridge** no experimento `base` (teste A, RMSE ponderado 1.915; logit base 1.916).

A escolha usa a validação B. O número do teste A é o holdout (Grupo A, múltiplos de 3) e não entra na escolha.

## 2018

Seções no recorte do logit base: Grupo A (teste) 121460, Grupo B (validação) 122323, Grupo C (treino) 122437. Peso: `QT_APTOS_t2`. UFs no agregado: 26.

Tempo de parede: 421.8 s.

### Experimento `base`

| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge ** | 2.424 | 1.891 | 0.982 | 95.24 | 2.419 | 2.409 | 2.420 | 2.417 | 2.468 | 0.000 |
| logit | 2.424 | 1.891 | 0.982 | 95.23 | 2.419 | 2.409 | 2.420 | 2.418 | 2.467 | 0.000 |
| hist_gradient_boosting | 2.482 | 1.932 | 0.981 | 94.88 | 2.471 | 2.197 | 2.471 | 2.472 | 2.924 | -0.058 |
| random_forest | 2.551 | 1.984 | 0.980 | 94.23 | 2.541 | 2.208 | 2.539 | 2.544 | 2.892 | -0.127 |
| fractional_logit | 2.559 | 2.001 | 0.980 | 94.08 | 2.553 | 2.540 | 2.553 | 2.548 | 2.913 | -0.135 |
| logit_ols | 2.624 | 2.046 | 0.979 | 93.58 | 2.617 | 2.605 | 2.616 | 2.614 | 3.260 | -0.200 |

Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.

### Experimento `extra`

| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting ** | 2.484 | 1.933 | 0.981 | 94.83 | 2.473 | 2.184 | 2.474 | 2.473 | — | 0.091 |
| random_forest | 2.550 | 1.984 | 0.980 | 94.29 | 2.540 | 2.202 | 2.540 | 2.545 | — | 0.025 |
| ridge | 2.549 | 1.974 | 0.980 | 94.37 | 2.546 | 2.409 | 2.602 | 2.572 | — | 0.025 |
| logit | 2.575 | 1.992 | 0.980 | 94.15 | 2.573 | 2.406 | 2.619 | 2.611 | — | 0.000 |
| fractional_logit | 2.782 | 2.149 | 0.976 | 92.45 | 2.770 | 2.536 | 2.753 | 2.659 | — | -0.207 |
| logit_ols | 3.123 | 2.305 | 0.970 | 90.86 | 3.102 | 2.603 | 2.856 | 2.753 | — | -0.548 |

Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.

### Experimento `extra_uf`

| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting ** | 2.633 | 2.057 | 0.978 | 93.67 | 2.624 | 2.597 | 2.623 | 2.624 | — | 0.226 |
| random_forest | 2.767 | 2.160 | 0.975 | 92.49 | 2.756 | 2.708 | 2.752 | 2.752 | — | 0.093 |
| logit | 2.859 | 2.245 | 0.974 | 91.49 | 2.856 | 2.719 | 2.911 | 2.778 | — | 0.000 |
| ridge | 2.859 | 2.245 | 0.974 | 91.49 | 2.856 | 2.719 | 2.911 | 2.778 | — | -0.000 |
| fractional_logit | 3.464 | 2.789 | 0.961 | 84.84 | 3.441 | 2.819 | 3.185 | 3.615 | — | -0.605 |
| logit_ols | 4.123 | 3.375 | 0.945 | 76.06 | 4.097 | 2.906 | 3.698 | 4.342 | — | -1.263 |

Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.

### Por UF (teste A, experimento base, RMSE ponderado)

| UF | n | logit | selecionado no base |
| --- | --- | --- | --- |
| AC | 434 | 2.932 | ridge 2.920 |
| AL | 1961 | 2.253 | ridge 2.253 |
| AM | 1882 | 2.293 | ridge 2.293 |
| AP | 481 | 2.706 | ridge 2.706 |
| BA | 5978 | 2.192 | ridge 2.192 |
| CE | 4168 | 2.734 | ridge 2.734 |
| DF | 2131 | 2.265 | ridge 2.262 |
| ES | 2360 | 2.470 | ridge 2.469 |
| GO | 4542 | 2.480 | ridge 2.479 |
| MA | 2339 | 2.233 | ridge 2.233 |
| MG | 13008 | 2.591 | ridge 2.591 |
| MS | 1893 | 2.564 | ridge 2.564 |
| MT | 2006 | 2.466 | ridge 2.466 |
| PA | 5206 | 2.711 | ridge 2.711 |
| PB | 2676 | 2.149 | ridge 2.149 |
| PE | 4887 | 2.062 | ridge 2.062 |
| PI | 1761 | 2.279 | ridge 2.279 |
| PR | 7652 | 2.420 | ridge 2.420 |
| RJ | 9711 | 2.251 | ridge 2.251 |
| RN | 1616 | 2.508 | ridge 2.508 |
| RO | 1216 | 2.594 | ridge 2.594 |
| RR | 345 | 2.650 | ridge 2.650 |
| RS | 8461 | 2.449 | ridge 2.450 |
| SC | 4685 | 2.354 | ridge 2.354 |
| SE | 1279 | 2.199 | ridge 2.200 |
| SP | 28782 | 2.452 | ridge 2.452 |

## 2022

Seções no recorte do logit base: Grupo A (teste) 91983, Grupo B (validação) 92591, Grupo C (treino) 92655. Peso: `QT_APTOS_t2`. UFs no agregado: 19.

Tempo de parede: 315.4 s.

### Experimento `base`

| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge ** | 1.821 | 1.427 | 0.986 | 98.84 | 1.813 | 1.825 | 1.814 | 1.831 | 1.886 | 0.000 |
| logit | 1.821 | 1.428 | 0.986 | 98.84 | 1.813 | 1.825 | 1.814 | 1.831 | 1.886 | 0.000 |
| hist_gradient_boosting | 1.878 | 1.464 | 0.985 | 98.62 | 1.867 | 1.695 | 1.869 | 1.889 | 2.046 | -0.056 |
| random_forest | 1.894 | 1.475 | 0.985 | 98.56 | 1.883 | 1.635 | 1.890 | 1.901 | 2.036 | -0.072 |
| fractional_logit | 2.078 | 1.626 | 0.981 | 97.61 | 2.078 | 2.080 | 2.078 | 2.086 | 2.237 | -0.257 |
| logit_ols | 2.194 | 1.705 | 0.979 | 96.94 | 2.194 | 2.195 | 2.190 | 2.199 | 2.423 | -0.372 |

Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.

### Experimento `extra`

| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting ** | 1.881 | 1.466 | 0.985 | 98.60 | 1.871 | 1.709 | 1.873 | 1.891 | — | 0.290 |
| random_forest | 1.895 | 1.476 | 0.985 | 98.56 | 1.885 | 1.629 | 1.890 | 1.901 | — | 0.276 |
| ridge | 2.165 | 1.631 | 0.980 | 97.12 | 2.154 | 1.824 | 2.067 | 2.443 | — | 0.007 |
| logit | 2.172 | 1.637 | 0.980 | 97.09 | 2.160 | 1.822 | 2.093 | 2.449 | — | 0.000 |
| fractional_logit | 2.818 | 1.963 | 0.966 | 94.09 | 2.806 | 2.077 | 3.425 | 3.772 | — | -0.647 |
| logit_ols | 3.044 | 2.106 | 0.960 | 92.84 | 3.031 | 2.193 | 3.544 | 4.128 | — | -0.873 |

Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.

### Experimento `extra_uf`

| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting ** | 1.867 | 1.463 | 0.986 | 98.68 | 1.865 | 1.853 | 1.869 | 1.879 | — | 0.083 |
| random_forest | 1.914 | 1.499 | 0.986 | 98.48 | 1.908 | 1.874 | 1.905 | 1.918 | — | 0.036 |
| ridge | 1.943 | 1.525 | 0.985 | 98.42 | 1.941 | 1.874 | 1.863 | 1.874 | — | 0.007 |
| logit | 1.950 | 1.530 | 0.985 | 98.39 | 1.946 | 1.871 | 1.863 | 1.874 | — | 0.000 |
| fractional_logit | 3.630 | 2.988 | 0.948 | 82.69 | 3.626 | 2.316 | 3.977 | 4.271 | — | -1.680 |
| logit_ols | 4.440 | 3.704 | 0.923 | 71.39 | 4.436 | 2.500 | 5.317 | 5.756 | — | -2.490 |

Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.

### Por UF (teste A, experimento base, RMSE ponderado)

| UF | n | logit | selecionado no base |
| --- | --- | --- | --- |
| AC | 593 | 2.405 | ridge 2.400 |
| AL | 2066 | 1.949 | ridge 1.949 |
| AM | 2168 | 2.140 | ridge 2.140 |
| AP | 565 | 2.387 | ridge 2.384 |
| BA | 11090 | 1.679 | ridge 1.679 |
| CE | 7033 | 1.774 | ridge 1.774 |
| DF | 2225 | 1.680 | ridge 1.677 |
| ES | 2976 | 1.861 | ridge 1.862 |
| GO | 4259 | 1.863 | ridge 1.862 |
| MA | 2855 | 1.693 | ridge 1.693 |
| MG | 16225 | 1.967 | ridge 1.967 |
| MS | 2222 | 2.093 | ridge 2.093 |
| MT | 2503 | 1.886 | ridge 1.886 |
| PA | 4742 | 1.935 | ridge 1.935 |
| PB | 2395 | 1.651 | ridge 1.651 |
| PE | 6771 | 1.497 | ridge 1.497 |
| PI | 1997 | 1.751 | ridge 1.750 |
| PR | 7984 | 1.752 | ridge 1.752 |
| RJ | 11314 | 1.801 | ridge 1.801 |

## 2026

Seções no recorte do logit base: Grupo A (teste) 156496, Grupo B (validação) 157532, Grupo C (treino) 157420. Peso: `QT_APTOS_t2`. UFs no agregado: 28.

Tempo de parede: 547.9 s.

### Experimento `base`

| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ridge ** | 1.915 | 1.502 | 0.985 | 98.58 | 1.916 | 1.914 | 1.916 | 1.920 | 1.968 | 0.001 |
| logit | 1.916 | 1.503 | 0.985 | 98.57 | 1.917 | 1.913 | 1.916 | 1.920 | 1.967 | 0.000 |
| hist_gradient_boosting | 1.972 | 1.537 | 0.984 | 98.34 | 1.972 | 1.777 | 1.980 | 1.983 | 2.150 | -0.056 |
| random_forest | 2.000 | 1.559 | 0.984 | 98.24 | 1.999 | 1.759 | 2.004 | 2.002 | 2.139 | -0.084 |
| fractional_logit | 2.042 | 1.602 | 0.983 | 97.94 | 2.045 | 2.040 | 2.045 | 2.047 | 2.154 | -0.126 |
| logit_ols | 2.096 | 1.640 | 0.982 | 97.64 | 2.099 | 2.094 | 2.097 | 2.099 | 2.262 | -0.180 |

Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.

### Experimento `extra`

| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting ** | 1.975 | 1.540 | 0.984 | 98.35 | 1.976 | 1.796 | 1.979 | 1.978 | — | 3.044 |
| random_forest | 1.999 | 1.558 | 0.984 | 98.24 | 1.999 | 1.755 | 2.001 | 2.002 | — | 3.020 |
| fractional_logit | 4.566 | 2.238 | 0.914 | 94.80 | 4.518 | 2.039 | 2.249 | 2.541 | — | 0.453 |
| logit_ols | 4.674 | 2.341 | 0.910 | 93.76 | 4.627 | 2.093 | 2.417 | 2.865 | — | 0.345 |
| ridge | 4.907 | 2.107 | 0.901 | 95.84 | 4.871 | 1.913 | 1.954 | 2.015 | — | 0.112 |
| logit | 5.019 | 2.131 | 0.896 | 95.77 | 4.983 | 1.911 | 2.005 | 2.136 | — | 0.000 |

Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.

### Experimento `extra_uf`

| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hist_gradient_boosting ** | 2.046 | 1.611 | 0.983 | 98.12 | 2.044 | 2.029 | 2.043 | 2.075 | — | 0.159 |
| random_forest | 2.077 | 1.635 | 0.982 | 97.97 | 2.075 | 2.051 | 2.075 | 2.079 | — | 0.128 |
| logit | 2.204 | 1.750 | 0.980 | 97.37 | 2.201 | 2.034 | 2.163 | 2.219 | — | 0.000 |
| ridge | 2.204 | 1.750 | 0.980 | 97.37 | 2.201 | 2.034 | 2.163 | 2.219 | — | -0.000 |
| fractional_logit | 2.362 | 1.877 | 0.977 | 96.25 | 2.367 | 2.207 | 2.390 | 2.336 | — | -0.157 |
| logit_ols | 2.692 | 2.169 | 0.970 | 93.74 | 2.697 | 2.282 | 2.736 | 2.610 | — | -0.488 |

Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.

### Por UF (teste A, experimento base, RMSE ponderado)

| UF | n | logit | selecionado no base |
| --- | --- | --- | --- |
| AC | 714 | 2.536 | ridge 2.449 |
| AL | 2185 | 1.908 | ridge 1.908 |
| AM | 2463 | 2.036 | ridge 2.036 |
| AP | 625 | 2.483 | ridge 2.474 |
| BA | 11443 | 1.818 | ridge 1.818 |
| CE | 6616 | 1.873 | ridge 1.873 |
| DF | 2311 | 1.777 | ridge 1.777 |
| ES | 3155 | 1.961 | ridge 1.961 |
| GO | 5049 | 2.075 | ridge 2.074 |
| MA | 3415 | 1.810 | ridge 1.813 |
| MG | 16964 | 2.051 | ridge 2.051 |
| MS | 2274 | 2.024 | ridge 2.024 |
| MT | 2611 | 1.888 | ridge 1.888 |
| PA | 5866 | 2.016 | ridge 2.016 |
| PB | 3144 | 1.708 | ridge 1.708 |
| PE | 6776 | 1.600 | ridge 1.600 |
| PI | 3131 | 1.968 | ridge 1.971 |
| PR | 8750 | 1.852 | ridge 1.852 |
| RJ | 12426 | 1.931 | ridge 1.931 |
| RN | 2454 | 1.768 | ridge 1.768 |
| RO | 1495 | 1.895 | ridge 1.895 |
| RR | 467 | 2.166 | ridge 2.124 |
| RS | 8682 | 1.934 | ridge 1.934 |
| SC | 5648 | 1.751 | ridge 1.751 |
| SE | 1886 | 1.788 | ridge 1.788 |
| SP | 34276 | 1.918 | ridge 1.918 |
| TO | 1240 | 2.080 | ridge 2.080 |
| ZZ | 430 | 2.552 | ridge 2.546 |

Arquivos: `benchmark_modelos_{ano}.csv` e, quando o teste A cabe num arquivo enxuto, `residual_ml_{ano}.parquet`.

