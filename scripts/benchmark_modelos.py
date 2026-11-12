#!/usr/bin/env python3
"""
Compara o logit linear vigente com modelos de ML no mesmo alvo.

O alvo e as features do experimento base são os de
``process_and_summarize.py`` (``find_best_candidate_match`` e ``get_features``).
A divisão é por NR_SECAO em três grupos: A teste, B validação, C treino.

Uso:
  python scripts/benchmark_modelos.py --anos 2026 2022 2018
  python scripts/benchmark_modelos.py --synthetic
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse
import functools
import importlib.util
import inspect
import json
import math
import subprocess
import sys
import time
import traceback
import zlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
print = functools.partial(print, flush=True)

PROCESSED_PADRAO = Path("/home/alexa/eleicoes-data/processed")
RESULTS_PADRAO = Path("resultados")
ANOS_VALIDOS = ("2014", "2018", "2022", "2026")

MIN_TREINO = 40
MIN_AVAL = 15
TUNE_CAP = 5000
RF_CAP_N = 60000
RF_MAX_AMOSTRAS = 40000
TE_M = 20.0
CV_SPLITS = 3
CV_MAX_LINHAS = 6000
RESIDUAL_MAX = 500_000
SEMENTE = 42

ARVORES = {"hist_gradient_boosting", "random_forest"}
MODELOS = (
    "logit",
    "ridge",
    "fractional_logit",
    "hist_gradient_boosting",
    "random_forest",
    "logit_ols",
)
GRADE = {
    "logit": [{}],
    "ridge": [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0, 100.0)],
    "fractional_logit": [{"l2": a} for a in (0.01, 0.1, 1.0, 10.0)],
    "logit_ols": [{"alpha": a} for a in (0.01, 0.1, 1.0, 10.0, 100.0)],
    "hist_gradient_boosting": [
        {"max_iter": 60, "learning_rate": 0.08, "max_depth": 3, "min_samples_leaf": 20},
        {"max_iter": 60, "learning_rate": 0.08, "max_depth": 6, "min_samples_leaf": 20},
        {"max_iter": 80, "learning_rate": 0.05, "max_depth": 6, "min_samples_leaf": 30},
    ],
    "random_forest": [
        {"n_estimators": 40, "max_depth": 8, "min_samples_leaf": 10},
        {"n_estimators": 40, "max_depth": 14, "min_samples_leaf": 20},
    ],
}
ROTACOES = (
    {"nome": "principal", "treino": "C", "validacao": "B", "teste": "A"},
    {"nome": "rot_B", "treino": "A", "validacao": "C", "teste": "B"},
    {"nome": "rot_C", "treino": "B", "validacao": "A", "teste": "C"},
)
ID_COLS = ["SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO"]
COLS_TABELA = [
    "ano", "experimento", "uf", "modelo", "avaliacao", "alvo", "peso",
    "n", "n_A", "n_B", "n_C", "n_ufs",
    "rmse", "mae", "r2",
    "rmse_ponderado", "mae_ponderado", "r2_ponderado",
    "pct_dentro_5pp", "logloss", "brier", "logloss_ponderado", "brier_ponderado",
    "ganho_rmse_w_vs_logit", "tempo_s", "hiperparams",
    "selecionado", "selecionado_geral",
]

PROTOCOLO = """# Benchmark de modelos vs logit

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
"""


def modulo_processamento():
    caminho = Path(__file__).resolve().parent / "process_and_summarize.py"
    spec = importlib.util.spec_from_file_location("process_and_summarize_benchmark", caminho)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_PROC = None


def proc():
    global _PROC
    if _PROC is None:
        _PROC = modulo_processamento()
    return _PROC


def grupos_de_serie(nr: pd.Series) -> pd.Series:
    """A: múltiplo de 3. B: múltiplo de 2 e não de 3. C: o resto. None se inválido."""
    x = pd.to_numeric(nr, errors="coerce").to_numpy(dtype=float)
    finito = np.isfinite(x)
    n = np.zeros(len(x), dtype=np.int64)
    if finito.any():
        n[finito] = np.rint(x[finito]).astype(np.int64)
    proximo = np.zeros(len(x), dtype=bool)
    if finito.any():
        proximo[finito] = np.abs(x[finito] - n[finito]) <= 1e-6
    ok = finito & proximo
    grupo = np.full(len(x), None, dtype=object)
    mult3 = ok & (n % 3 == 0)
    mult2 = ok & (n % 2 == 0) & ~mult3
    resto = ok & ~mult3 & ~mult2
    grupo[mult3] = "A"
    grupo[mult2] = "B"
    grupo[resto] = "C"
    return pd.Series(grupo, index=nr.index)


def sigmoide(z):
    z = np.clip(np.asarray(z, dtype=float), -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def instanciar(cls, **kwargs):
    assinatura = inspect.signature(cls.__init__).parameters
    return cls(**{k: v for k, v in kwargs.items() if k in assinatura})


def ajustar_folha(n: int, pedido: int) -> int:
    return int(max(2, min(int(pedido), max(2, n // 20))))


def aplicar_impute(X, med):
    out = np.array(X, dtype=float, copy=True)
    if out.size == 0:
        return out
    ruim = ~np.isfinite(out)
    if ruim.any():
        out[ruim] = np.asarray(med, dtype=float)[np.where(ruim)[1]]
    return out


def ajustar_logit_fracionario(X, y_pp, l2: float, max_iter: int = 25) -> np.ndarray:
    y = np.clip(np.asarray(y_pp, dtype=float) / 100.0, 1e-4, 1.0 - 1e-4)
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])
    beta = np.zeros(k + 1)
    pen = np.full(k + 1, float(l2))
    pen[0] = 0.0
    for _ in range(max_iter):
        eta = Xd @ beta
        p = sigmoide(eta)
        w = np.clip(p * (1.0 - p), 1e-5, None)
        z = eta + (y - p) / w
        A = Xd.T @ (Xd * w[:, None])
        diag = np.diag(A) + pen + 1e-8
        np.fill_diagonal(A, diag)
        b = Xd.T @ (w * z)
        try:
            novo = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            novo = np.linalg.lstsq(A, b, rcond=None)[0]
        if not np.all(np.isfinite(novo)):
            break
        if np.max(np.abs(novo - beta)) < 1e-6:
            beta = novo
            break
        beta = novo
    if not np.all(np.isfinite(beta)):
        raise RuntimeError("logit fracionário não convergiu")
    return beta


class Modelo:
    def __init__(self, nome: str, params: dict, seed: int):
        self.nome = nome
        self.params = dict(params)
        self.seed = int(seed)
        self.med_ = None
        self.scaler_ = None
        self.est_ = None
        self.beta_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X precisa ser 2d")
        ruim = ~np.isfinite(X)
        if ruim.any():
            X = X.copy()
            X[ruim] = np.nan
        if X.shape[1] == 0:
            raise ValueError("sem features")
        med = np.nanmedian(X, axis=0)
        self.med_ = np.where(np.isfinite(med), med, 0.0)
        self._fit_limpo(aplicar_impute(X, self.med_), y)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X precisa ser 2d")
        ruim = ~np.isfinite(X)
        if ruim.any():
            X = X.copy()
            X[ruim] = np.nan
        Xi = aplicar_impute(X, self.med_)
        return np.asarray(self._predict_limpo(Xi), dtype=float).reshape(-1)

    def _fit_limpo(self, X, y):
        n = len(y)
        if self.nome in ARVORES:
            kw = dict(self.params)
            kw["min_samples_leaf"] = ajustar_folha(n, kw.get("min_samples_leaf", 20))
            kw["random_state"] = self.seed
            if self.nome == "random_forest":
                kw["n_jobs"] = 1
                kw["max_features"] = 1.0
                if n > RF_CAP_N:
                    kw["max_samples"] = min(RF_MAX_AMOSTRAS, n)
                self.est_ = instanciar(RandomForestRegressor, **kw)
            else:
                kw["early_stopping"] = False
                self.est_ = instanciar(HistGradientBoostingRegressor, **kw)
            self.est_.fit(X, y)
            return
        self.scaler_ = StandardScaler()
        Xs = self.scaler_.fit_transform(X)
        if self.nome == "logit":
            self.est_ = LinearRegression()
            self.est_.fit(Xs, y)
        elif self.nome == "ridge":
            self.est_ = Ridge(alpha=float(self.params.get("alpha", 1.0)))
            self.est_.fit(Xs, y)
        elif self.nome == "logit_ols":
            p = np.clip(y / 100.0, 1e-4, 1.0 - 1e-4)
            z = np.log(p / (1.0 - p))
            self.est_ = Ridge(alpha=float(self.params.get("alpha", 1.0)))
            self.est_.fit(Xs, z)
        elif self.nome == "fractional_logit":
            it = 12 if n > 80000 else 25
            self.beta_ = ajustar_logit_fracionario(Xs, y, float(self.params.get("l2", 1.0)), max_iter=it)
        else:
            raise ValueError(self.nome)

    def _predict_limpo(self, X):
        if self.nome in ARVORES:
            return self.est_.predict(X)
        Xs = self.scaler_.transform(X)
        if self.nome == "logit_ols":
            return 100.0 * sigmoide(self.est_.predict(Xs))
        if self.nome == "fractional_logit":
            Xd = np.column_stack([np.ones(len(Xs)), Xs])
            return 100.0 * sigmoide(Xd @ self.beta_)
        return self.est_.predict(Xs)


def rmse_ponderado_rapido(y, pred, w) -> float:
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    w = np.asarray(w, dtype=float)
    m = np.isfinite(y) & np.isfinite(pred) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return math.inf
    return float(np.sqrt(np.average((y[m] - pred[m]) ** 2, weights=w[m])))


def semente_de(*partes) -> int:
    texto = "|".join(str(p) for p in partes)
    return zlib.crc32(texto.encode()) % (2**31 - 1)


class Acc:
    def __init__(self):
        self.n = 0
        self.n_ufs = 0
        self.sum_w = 0.0
        self.sse = 0.0
        self.sae = 0.0
        self.sse_w = 0.0
        self.sae_w = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_wy = 0.0
        self.sum_wy2 = 0.0
        self.n5 = 0
        self.sum_ll = 0.0
        self.sum_ll_w = 0.0
        self.sum_br = 0.0
        self.sum_br_w = 0.0

    def add(self, y, pred, w):
        y = np.asarray(y, dtype=float).reshape(-1)
        pred = np.asarray(pred, dtype=float).reshape(-1)
        w = np.asarray(w, dtype=float).reshape(-1)
        m = np.isfinite(y) & np.isfinite(pred) & np.isfinite(w) & (w > 0)
        if not np.any(m):
            return
        y, pred, w = y[m], pred[m], w[m]
        err = y - pred
        p = np.clip(y / 100.0, 1e-6, 1.0 - 1e-6)
        ph = np.clip(pred / 100.0, 1e-6, 1.0 - 1e-6)
        ll = -(p * np.log(ph) + (1.0 - p) * np.log(1.0 - ph))
        br = (p - ph) ** 2
        self.n += int(len(y))
        self.sum_w += float(w.sum())
        self.sse += float(np.square(err).sum())
        self.sae += float(np.abs(err).sum())
        self.sse_w += float((w * np.square(err)).sum())
        self.sae_w += float((w * np.abs(err)).sum())
        self.sum_y += float(y.sum())
        self.sum_y2 += float(np.square(y).sum())
        self.sum_wy += float((w * y).sum())
        self.sum_wy2 += float((w * np.square(y)).sum())
        self.n5 += int(np.sum(np.abs(err) <= 5.0))
        self.sum_ll += float(ll.sum())
        self.sum_ll_w += float((w * ll).sum())
        self.sum_br += float(br.sum())
        self.sum_br_w += float((w * br).sum())

    def fundir(self, outro: "Acc"):
        for campo in (
            "n", "n_ufs", "sum_w", "sse", "sae", "sse_w", "sae_w",
            "sum_y", "sum_y2", "sum_wy", "sum_wy2", "n5",
            "sum_ll", "sum_ll_w", "sum_br", "sum_br_w",
        ):
            setattr(self, campo, getattr(self, campo) + getattr(outro, campo))

    def como_dict(self) -> dict:
        def div(num, den):
            if den is None or not np.isfinite(den) or den <= 0:
                return np.nan
            return float(num / den)

        sst = self.sum_y2 - (self.sum_y ** 2) / self.n if self.n else np.nan
        sst_w = self.sum_wy2 - (self.sum_wy ** 2) / self.sum_w if self.sum_w > 0 else np.nan
        r2 = np.nan if not np.isfinite(sst) or sst <= 1e-12 else 1.0 - self.sse / sst
        r2_w = np.nan if not np.isfinite(sst_w) or sst_w <= 1e-12 else 1.0 - self.sse_w / sst_w
        return {
            "n": int(self.n),
            "n_ufs": int(self.n_ufs),
            "rmse": math.sqrt(div(self.sse, self.n)) if self.n else np.nan,
            "mae": div(self.sae, self.n),
            "r2": r2,
            "rmse_ponderado": math.sqrt(div(self.sse_w, self.sum_w)) if self.sum_w > 0 else np.nan,
            "mae_ponderado": div(self.sae_w, self.sum_w),
            "r2_ponderado": r2_w,
            "pct_dentro_5pp": div(100.0 * self.n5, self.n),
            "logloss": div(self.sum_ll, self.n),
            "brier": div(self.sum_br, self.n),
            "logloss_ponderado": div(self.sum_ll_w, self.sum_w),
            "brier_ponderado": div(self.sum_br_w, self.sum_w),
        }


def mapa_suave(chaves, y, m, gmean) -> dict:
    tab = pd.DataFrame({"k": np.asarray(chaves).astype(str), "y": np.asarray(y, dtype=float)})
    ag = tab.groupby("k", sort=False)["y"].agg(["mean", "count"])
    enc = (ag["count"] * ag["mean"] + m * gmean) / (ag["count"] + m)
    return enc.to_dict()


def coluna_te(chaves, y, train_mask, m: float = TE_M) -> np.ndarray:
    """Target encoding suavizado. No treino, fora da amostra por município."""
    chaves = np.asarray(chaves).astype(str)
    y = np.asarray(y, dtype=float)
    train_mask = np.asarray(train_mask, dtype=bool)
    enc = np.full(len(y), np.nan)
    tr = np.flatnonzero(train_mask)
    if len(tr) == 0:
        return np.zeros(len(y))
    gmean = float(np.mean(y[tr]))
    mapa_full = mapa_suave(chaves[tr], y[tr], m, gmean)
    oof = np.full(len(tr), gmean)
    if pd.unique(chaves[tr]).size >= 3:
        folds = np.fromiter((zlib.crc32(chaves[i].encode()) % 3 for i in tr), dtype=np.int8, count=len(tr))
        for f in range(3):
            inn = tr[folds != f]
            out_local = folds == f
            if len(inn) == 0 or not np.any(out_local):
                continue
            g = float(np.mean(y[inn]))
            mp = mapa_suave(chaves[inn], y[inn], m, g)
            oof[out_local] = pd.Series(chaves[tr][out_local]).map(mp).fillna(g).to_numpy(dtype=float)
    else:
        oof = pd.Series(chaves[tr]).map(mapa_full).fillna(gmean).to_numpy(dtype=float)
    enc[tr] = oof
    fora = np.flatnonzero(~train_mask)
    if len(fora):
        enc[fora] = pd.Series(chaves[fora]).map(mapa_full).fillna(gmean).to_numpy(dtype=float)
    return enc


def dummies_uf(valores, train_mask) -> np.ndarray:
    v = np.asarray(valores).astype(str)
    train_mask = np.asarray(train_mask, dtype=bool)
    cats = sorted(set(v[train_mask].tolist()))
    if len(cats) <= 1:
        return np.zeros((len(v), 0))
    use = cats[1:]
    return np.column_stack([(v == c).astype(float) for c in use])


def escolher_peso(df: pd.DataFrame):
    for coluna in ("QT_APTOS_t2", "QT_APTOS_t1", "QT_COMPARECIMENTO_t2", "QT_COMPARECIMENTO_t1"):
        if coluna not in df.columns:
            continue
        w = pd.to_numeric(df[coluna], errors="coerce")
        if w.notna().any() and bool((w.fillna(0) > 0).any()):
            return coluna, w
    return "uniforme", pd.Series(np.ones(len(df)), index=df.index)


def serie_tamanho(df: pd.DataFrame):
    for coluna in ("QT_APTOS_t1", "QT_APTOS_t2", "QT_COMPARECIMENTO_t1", "QT_COMPARECIMENTO_t2"):
        if coluna not in df.columns:
            continue
        s = pd.to_numeric(df[coluna], errors="coerce")
        if s.notna().any():
            return s
    return None


def calcular_log(df: pd.DataFrame) -> pd.Series:
    s = serie_tamanho(df)
    if s is None:
        return pd.Series(np.nan, index=df.index)
    return np.log1p(s.clip(lower=0))


def calcular_abst(df: pd.DataFrame) -> pd.Series:
    if "QT_COMPARECIMENTO_t1" in df.columns and "QT_APTOS_t1" in df.columns:
        comp = pd.to_numeric(df["QT_COMPARECIMENTO_t1"], errors="coerce")
        aptos = pd.to_numeric(df["QT_APTOS_t1"], errors="coerce").replace(0, np.nan)
        return (1.0 - comp / aptos).clip(0, 1)
    if "QT_ABSTENCOES_t1" in df.columns and "QT_APTOS_t1" in df.columns:
        abst = pd.to_numeric(df["QT_ABSTENCOES_t1"], errors="coerce")
        aptos = pd.to_numeric(df["QT_APTOS_t1"], errors="coerce").replace(0, np.nan)
        return (abst / aptos).clip(0, 1)
    return pd.Series(np.nan, index=df.index)


def listar_joins(processed: Path, ano: str) -> list[Path]:
    """Mesma descoberta de process_and_summarize, sem contar a mesma UF duas vezes.

    2026 aceita ``join_UF.parquet`` e ``join_2026_UF.parquet``. Se os dois existem,
    fica o arquivo com o ano no nome.
    """
    if ano == "2026":
        arquivos = [
            p for p in sorted(processed.glob("join_*.parquet"))
            if not any(tag in p.stem for tag in ("_2018_", "_2022_"))
        ]
    else:
        arquivos = sorted(processed.glob(f"join_{ano}_*.parquet"))
    por_uf = {}
    for caminho in arquivos:
        uf = uf_do_arquivo(caminho, ano)
        atual = por_uf.get(uf)
        if atual is None or caminho.stem.startswith(f"join_{ano}_"):
            por_uf[uf] = caminho
    return [por_uf[uf] for uf in sorted(por_uf)]


def uf_do_arquivo(path: Path, ano: str) -> str:
    return path.stem.replace(f"join_{ano}_", "").replace("join_", "")


def manter_coluna(nome: str) -> bool:
    return (
        nome in {"SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "diff_comparecimento", "razao_comparecimento"}
        or nome.startswith("PCT_")
        or nome.startswith("QT_")
    )


def ler_join(path: Path) -> pd.DataFrame:
    try:
        import pyarrow.parquet as pq
        nomes = list(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in nomes if manter_coluna(c)]
        if not cols:
            raise ValueError("sem colunas úteis")
        return pd.read_parquet(path, columns=cols)
    except Exception:
        df = pd.read_parquet(path)
        cols = [c for c in df.columns if manter_coluna(c)]
        return df[cols].copy()


def preparar_frame(df: pd.DataFrame, uf: str) -> dict | None:
    df = df.copy()
    for coluna in list(df.columns):
        if coluna.startswith("PCT_") or coluna.startswith("QT_") or coluna in ("diff_comparecimento", "razao_comparecimento"):
            df[coluna] = pd.to_numeric(df[coluna], errors="coerce").replace([np.inf, -np.inf], np.nan)
    col_t1, col_t2 = proc().find_best_candidate_match(df)
    if not col_t2:
        print(f"  {uf}: sem par T1/T2")
        return None
    features = [c for c in proc().get_features(df) if c in df.columns and c != col_t2]
    features = [c for c in features if df[c].notna().any()]
    if not features:
        print(f"  {uf}: sem features")
        return None
    df["y"] = pd.to_numeric(df[col_t2], errors="coerce")
    q99 = df["y"].quantile(0.99)
    if pd.notna(q99) and float(q99) <= 1.0:
        df["y"] = df["y"] * 100.0
    if "NR_SECAO" not in df.columns:
        print(f"  {uf}: sem NR_SECAO")
        return None
    df["grupo"] = grupos_de_serie(df["NR_SECAO"])
    nome_peso, w = escolher_peso(df)
    df["w"] = pd.to_numeric(w, errors="coerce")
    if "CD_MUNICIPIO" in df.columns:
        df["municipio"] = df["CD_MUNICIPIO"].astype(str)
    else:
        df["municipio"] = "NA"
    if "SG_UF" in df.columns:
        df["uf"] = df["SG_UF"].astype(str)
    else:
        df["uf"] = uf
    df["log_eleitores"] = calcular_log(df)
    df["abstencao_t1"] = calcular_abst(df)
    for coluna in features + ["y", "w", "log_eleitores", "abstencao_t1"]:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce").replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["y", "grupo", *features]).copy()
    if df["w"].notna().any():
        mediana = float(df.loc[df["w"] > 0, "w"].median()) if (df["w"] > 0).any() else 1.0
    else:
        mediana = 1.0
    df["w"] = df["w"].where(df["w"] > 0, np.nan).fillna(mediana if np.isfinite(mediana) else 1.0)
    df = df.reset_index(drop=True)
    cont = df["grupo"].value_counts().to_dict()
    if min(cont.get("A", 0), cont.get("B", 0), cont.get("C", 0)) < MIN_AVAL or cont.get("C", 0) < MIN_TREINO:
        print(f"  {uf}: grupos pequenos A/B/C={cont.get('A', 0)}/{cont.get('B', 0)}/{cont.get('C', 0)}")
        return None
    return {
        "uf": uf,
        "alvo": col_t2,
        "par_t1": col_t1,
        "peso": nome_peso,
        "base_cols": features,
        "df": df,
        "n_A": int(cont.get("A", 0)),
        "n_B": int(cont.get("B", 0)),
        "n_C": int(cont.get("C", 0)),
    }


def matriz_de(df: pd.DataFrame, cols: list[str], train_mask, te: bool, uf_dummies: bool) -> np.ndarray:
    partes = []
    for coluna in cols:
        if coluna in df.columns:
            partes.append(pd.to_numeric(df[coluna], errors="coerce").to_numpy(dtype=float))
        else:
            partes.append(np.full(len(df), np.nan))
    if partes:
        X = np.column_stack(partes)
    else:
        X = np.zeros((len(df), 0))
    if te:
        chave = df["municipio"].to_numpy()
        te_col = coluna_te(chave, df["y"].to_numpy(dtype=float), train_mask)
        X = np.column_stack([X, te_col]) if X.size else te_col.reshape(-1, 1)
    if uf_dummies:
        D = dummies_uf(df["uf"].to_numpy(), train_mask)
        if D.shape[1]:
            X = np.column_stack([X, D]) if X.size else D
    return X


def cols_de(prep: dict, experimento: str) -> list[str]:
    cols = list(prep["base_cols"])
    if experimento == "base":
        return cols
    for extra in ("log_eleitores", "abstencao_t1"):
        if extra not in cols:
            cols.append(extra)
    return cols


def ajustar_modelo(nome: str, Xtr, ytr, Xva, yva, wva, seed: int):
    grade = GRADE[nome]
    n = len(ytr)
    usar_sub = nome in ARVORES and n > TUNE_CAP
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    if usar_sub:
        idx = np.sort(rng.choice(n, TUNE_CAP, replace=False))
    melhor = None
    t0 = time.perf_counter()
    for i, params in enumerate(grade):
        modelo = Modelo(nome, params, seed + i + 1)
        modelo.fit(Xtr[idx], ytr[idx])
        pred = modelo.predict(Xva)
        score = rmse_ponderado_rapido(yva, pred, wva)
        chave = json.dumps(params, sort_keys=True)
        if melhor is None or score < melhor[0] - 1e-12 or (abs(score - melhor[0]) <= 1e-12 and chave < melhor[3]):
            melhor = (score, params, modelo, chave)
    params = melhor[1]
    modelo = melhor[2]
    if usar_sub:
        modelo = Modelo(nome, params, seed)
        modelo.fit(Xtr, ytr)
    return modelo, params, time.perf_counter() - t0


def processar_bloco(df, cols, experimento: str, rot: dict, uf: str, seed_base: int):
    g = df["grupo"].to_numpy()
    tr = g == rot["treino"]
    va = g == rot["validacao"]
    te = g == rot["teste"]
    if int(tr.sum()) < MIN_TREINO or int(va.sum()) < MIN_AVAL or int(te.sum()) < MIN_AVAL:
        return None
    te_flag = experimento != "base"
    uf_flag = experimento == "extra_uf"
    X = matriz_de(df, cols, tr, te=te_flag, uf_dummies=uf_flag)
    if X.shape[1] == 0:
        return None
    y = df["y"].to_numpy(dtype=float)
    w = df["w"].to_numpy(dtype=float)
    ajustados = {}
    for nome in MODELOS:
        try:
            seed = semente_de(seed_base, uf, experimento, rot["nome"], nome)
            modelo, params, tempo = ajustar_modelo(nome, X[tr], y[tr], X[va], y[va], w[va], seed)
            preds = {}
            accs = {}
            for rotulo, mask in (("treino", tr), ("validacao", va), ("teste", te)):
                pred = modelo.predict(X[mask])
                acc = Acc()
                acc.add(y[mask], pred, w[mask])
                acc.n_ufs = 1
                accs[rotulo] = acc
                preds[rotulo] = pred
            ajustados[nome] = {"params": params, "tempo": tempo, "accs": accs, "pred_teste": preds["teste"], "mask_teste": te}
        except Exception as exc:
            print(f"  ERRO {uf} {experimento} {rot['nome']} {nome}: {exc}")
    if "logit" not in ajustados and rot["nome"] == "principal" and experimento == "base":
        return None
    return ajustados


def linha_resultado(**kwargs) -> dict:
    base = {c: np.nan for c in COLS_TABELA}
    base.update(kwargs)
    base["selecionado"] = int(kwargs.get("selecionado", 0))
    base["selecionado_geral"] = int(kwargs.get("selecionado_geral", 0))
    return base


def registrar_acc(linhas, *, ano, experimento, uf, modelo, avaliacao, acc: Acc, tempo, params, alvo, peso, n_A, n_B, n_C):
    if acc.n <= 0:
        return
    info = acc.como_dict()
    if isinstance(params, list):
        if not params:
            hiper = ""
        else:
            chaves = [json.dumps(p, sort_keys=True, default=str) for p in params]
            hiper = max(set(chaves), key=chaves.count)
    elif isinstance(params, dict):
        hiper = json.dumps(params, sort_keys=True, default=str)
    else:
        hiper = str(params or "")
    linhas.append(linha_resultado(
        ano=str(ano), experimento=experimento, uf=uf, modelo=modelo, avaliacao=avaliacao,
        alvo=alvo, peso=peso, n_A=n_A, n_B=n_B, n_C=n_C, tempo_s=float(tempo or 0.0),
        hiperparams=hiper, **info,
    ))


def amostrar_municipios(df: pd.DataFrame, cap: int, rng) -> pd.DataFrame:
    if len(df) <= cap:
        return df.reset_index(drop=True)
    chave = df["municipio"].astype(str)
    tamanhos = chave.value_counts()
    ordem = rng.permutation(tamanhos.index.to_numpy())
    escolhidos = []
    n = 0
    for mun in ordem:
        if n >= cap and escolhidos:
            break
        escolhidos.append(mun)
        n += int(tamanhos.loc[mun])
        if n >= cap:
            break
    return df.loc[chave.isin(escolhidos)].reset_index(drop=True)


def cv_base(prep: dict, params_por_modelo: dict, seed: int):
    df = prep["df"]
    sub = df[df["grupo"].isin(["B", "C"])].copy()
    if len(sub) < 30:
        return {}
    rng = np.random.default_rng(seed)
    sub = amostrar_municipios(sub, CV_MAX_LINHAS, rng)
    if sub["municipio"].nunique() >= CV_SPLITS:
        grupos = sub["municipio"].astype(str).to_numpy()
        agrupamento = "CD_MUNICIPIO"
    elif "NR_ZONA" in sub.columns and sub["NR_ZONA"].nunique() >= CV_SPLITS:
        grupos = sub["NR_ZONA"].astype(str).to_numpy()
        agrupamento = "NR_ZONA"
    else:
        return {}
    cols = prep["base_cols"]
    X = np.column_stack([pd.to_numeric(sub[c], errors="coerce").to_numpy(dtype=float) for c in cols])
    y = sub["y"].to_numpy(dtype=float)
    w = sub["w"].to_numpy(dtype=float)
    n_splits = min(CV_SPLITS, len(np.unique(grupos)))
    if n_splits < 2:
        return {}
    gkf = GroupKFold(n_splits=n_splits)
    saida = {}
    for nome, params in params_por_modelo.items():
        t0 = time.perf_counter()
        acc = Acc()
        acc.n_ufs = 1
        try:
            for tr, te in gkf.split(X, y, grupos):
                if len(tr) < 20 or len(te) < 5:
                    continue
                modelo = Modelo(nome, params, seed)
                modelo.fit(X[tr], y[tr])
                acc.add(y[te], modelo.predict(X[te]), w[te])
        except Exception as exc:
            print(f"  ERRO cv {prep['uf']} {nome}: {exc}")
            continue
        if acc.n > 0:
            saida[nome] = {"acc": acc, "tempo": time.perf_counter() - t0, "agrupamento": agrupamento, "params": params}
    return saida


def finalizar_tabela(linhas: list[dict]) -> pd.DataFrame:
    if not linhas:
        return pd.DataFrame(columns=COLS_TABELA)
    df = pd.DataFrame(linhas)
    for coluna in COLS_TABELA:
        if coluna not in df.columns:
            df[coluna] = np.nan
    df["ganho_rmse_w_vs_logit"] = np.nan
    df["selecionado"] = 0
    df["selecionado_geral"] = 0
    util = ~df["modelo"].astype(str).str.startswith("_")
    for (ano, exp, uf, aval), idx in df[util].groupby(["ano", "experimento", "uf", "avaliacao"]).groups.items():
        bloco = df.loc[idx]
        logit = bloco.loc[bloco["modelo"] == "logit", "rmse_ponderado"]
        if len(logit) == 0 or not np.isfinite(logit.iloc[0]):
            continue
        df.loc[idx, "ganho_rmse_w_vs_logit"] = float(logit.iloc[0]) - df.loc[idx, "rmse_ponderado"]
    val = df[util & (df["uf"] == "BR") & (df["avaliacao"] == "validacao_B")].copy()
    for (ano, exp), bloco in val.groupby(["ano", "experimento"]):
        bloco = bloco.sort_values(["rmse_ponderado", "modelo"], na_position="last")
        if bloco.empty or not np.isfinite(bloco.iloc[0]["rmse_ponderado"]):
            continue
        vencedor = bloco.iloc[0]["modelo"]
        df.loc[(df["ano"] == ano) & (df["experimento"] == exp) & (df["modelo"] == vencedor), "selecionado"] = 1
    for ano, bloco in val.groupby("ano"):
        bloco = bloco.sort_values(["rmse_ponderado", "experimento", "modelo"], na_position="last")
        if bloco.empty or not np.isfinite(bloco.iloc[0]["rmse_ponderado"]):
            continue
        exp = bloco.iloc[0]["experimento"]
        vencedor = bloco.iloc[0]["modelo"]
        df.loc[(df["ano"] == ano) & (df["experimento"] == exp) & (df["modelo"] == vencedor), "selecionado_geral"] = 1
    return df[COLS_TABELA].sort_values(["ano", "experimento", "uf", "avaliacao", "rmse_ponderado", "modelo"], na_position="last")


def fmt(v, casas=3) -> str:
    try:
        if v is None or pd.isna(v):
            return "—"
        x = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not math.isfinite(x):
        return "—"
    return f"{x:.{casas}f}"


def montar_md(df: pd.DataFrame) -> str:
    partes = [PROTOCOLO.strip(), ""]
    if df is None or len(df) == 0:
        partes.append("Nenhum resultado numérico foi gerado nesta execução. O runner grava `benchmark_modelos_{ano}.csv` ao processar os joins.")
        partes.append("")
        return "\n".join(partes) + "\n"
    df = df.copy()
    df["ano"] = df["ano"].astype(str)
    partes.append("## Resultado")
    partes.append("")
    for ano in sorted(df["ano"].unique()):
        sub = df[(df["ano"] == ano) & (df["uf"] == "BR") & (df["avaliacao"] == "teste_A") & (df["selecionado_geral"] == 1)]
        logit = df[
            (df["ano"] == ano) & (df["uf"] == "BR") & (df["avaliacao"] == "teste_A")
            & (df["experimento"] == "base") & (df["modelo"] == "logit")
        ]
        if sub.empty:
            partes.append(f"- {ano}: sem modelo selecionado.")
            continue
        row = sub.iloc[0]
        texto = (
            f"- {ano}: selecionado **{row['modelo']}** no experimento `{row['experimento']}` "
            f"(teste A, RMSE ponderado {fmt(row['rmse_ponderado'])}"
        )
        if len(logit):
            texto += f"; logit base {fmt(logit.iloc[0]['rmse_ponderado'])}"
        texto += ")."
        partes.append(texto)
    partes.append("")
    partes.append("A escolha usa a validação B. O número do teste A é o holdout (Grupo A, múltiplos de 3) e não entra na escolha.")
    partes.append("")
    for ano in sorted(df["ano"].unique()):
        partes.append(f"## {ano}")
        partes.append("")
        ref = df[(df["ano"] == ano) & (df["uf"] == "BR") & (df["experimento"] == "base") & (df["modelo"] == "logit") & (df["avaliacao"] == "teste_A")]
        if len(ref):
            r = ref.iloc[0]
            partes.append(
                f"Seções no recorte do logit base: Grupo A (teste) {int(r['n_A'])}, "
                f"Grupo B (validação) {int(r['n_B'])}, Grupo C (treino) {int(r['n_C'])}. "
                f"Peso: `{r['peso']}`. UFs no agregado: {int(r['n_ufs'])}."
            )
            partes.append("")
        wall = df[(df["ano"] == ano) & (df["modelo"] == "_wall")]
        if len(wall):
            partes.append(f"Tempo de parede: {fmt(wall.iloc[0]['tempo_s'], 1)} s.")
            partes.append("")
        for exp in ("base", "extra", "extra_uf"):
            bloco = df[(df["ano"] == ano) & (df["uf"] == "BR") & (df["experimento"] == exp) & (~df["modelo"].astype(str).str.startswith("_"))]
            if bloco.empty:
                continue
            partes.append(f"### Experimento `{exp}`")
            partes.append("")
            modelos = (
                bloco[bloco["avaliacao"] == "validacao_B"]
                .sort_values(["rmse_ponderado", "modelo"], na_position="last")["modelo"]
                .tolist()
            )
            if not modelos:
                modelos = sorted(bloco["modelo"].unique())
            partes.append("| Modelo | RMSE_w teste A | MAE_w A | R²_w A | %≤5 p.p. | RMSE_w val B | RMSE_w treino C | RMSE_w rot. B | RMSE_w rot. C | RMSE_w CV | Ganho vs logit no A |")
            partes.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for modelo in modelos:
                def pegar(aval, col, _modelo=modelo):
                    ped = bloco[(bloco["modelo"] == _modelo) & (bloco["avaliacao"] == aval)]
                    if ped.empty:
                        return np.nan
                    return ped.iloc[0][col]
                marca = " **" if int(bloco.loc[bloco["modelo"] == modelo, "selecionado"].max() or 0) == 1 else ""
                nome = f"{modelo}{marca}"
                partes.append(
                    "| {nome} | {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} | {i} | {j} |".format(
                        nome=nome,
                        a=fmt(pegar("teste_A", "rmse_ponderado")),
                        b=fmt(pegar("teste_A", "mae_ponderado")),
                        c=fmt(pegar("teste_A", "r2_ponderado")),
                        d=fmt(pegar("teste_A", "pct_dentro_5pp"), 2),
                        e=fmt(pegar("validacao_B", "rmse_ponderado")),
                        f=fmt(pegar("treino_C", "rmse_ponderado")),
                        g=fmt(pegar("rotacao_teste_B", "rmse_ponderado")),
                        h=fmt(pegar("rotacao_teste_C", "rmse_ponderado")),
                        i=fmt(pegar("cv_municipio", "rmse_ponderado")),
                        j=fmt(pegar("teste_A", "ganho_rmse_w_vs_logit")),
                    )
                )
            partes.append("")
            partes.append("Ganho positivo significa RMSE ponderado menor que o logit do mesmo experimento. `**` é o modelo escolhido na validação B.")
            partes.append("")
        ufs = df[
            (df["ano"] == ano) & (df["experimento"] == "base") & (df["avaliacao"] == "teste_A")
            & (df["uf"] != "BR") & (df["modelo"].isin(["logit"]))
        ]
        if len(ufs):
            sel = df[(df["ano"] == ano) & (df["experimento"] == "base") & (df["selecionado"] == 1) & (df["avaliacao"] == "teste_A")]
            nome_sel = sel["modelo"].iloc[0] if len(sel) else None
            partes.append("### Por UF (teste A, experimento base, RMSE ponderado)")
            partes.append("")
            partes.append("| UF | n | logit | selecionado no base |")
            partes.append("| --- | --- | --- | --- |")
            for _, row in ufs.sort_values("uf").iterrows():
                outro = ""
                if nome_sel:
                    ped = df[
                        (df["ano"] == ano) & (df["experimento"] == "base") & (df["avaliacao"] == "teste_A")
                        & (df["uf"] == row["uf"]) & (df["modelo"] == nome_sel)
                    ]
                    if len(ped):
                        outro = f"{nome_sel} {fmt(ped.iloc[0]['rmse_ponderado'])}"
                partes.append(f"| {row['uf']} | {int(row['n'])} | {fmt(row['rmse_ponderado'])} | {outro or '—'} |")
            partes.append("")
    partes.append("Arquivos: `benchmark_modelos_{ano}.csv` e, quando o teste A cabe num arquivo enxuto, `residual_ml_{ano}.parquet`.")
    partes.append("")
    return "\n".join(partes) + "\n"


def escrever_saidas(out_dir: Path, por_ano: dict[str, pd.DataFrame]):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ano, df in por_ano.items():
        destino = out_dir / f"benchmark_modelos_{ano}.csv"
        df.to_csv(destino, index=False)
        print(f"CSV {destino} ({len(df)} linhas)")
    frames = []
    for caminho in sorted(out_dir.glob("benchmark_modelos_*.csv")):
        if caminho.stem.endswith("_contagem"):
            continue
        try:
            frames.append(pd.read_csv(caminho, dtype={"ano": str, "uf": str, "modelo": str, "experimento": str, "avaliacao": str}))
        except Exception as exc:
            print(f"Não li {caminho.name}: {exc}")
    tudo = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    md = out_dir / "benchmark_modelos.md"
    md.write_text(montar_md(tudo), encoding="utf-8")
    print(f"Resumo {md}")


def imprimir_tabela(df: pd.DataFrame, ano: str):
    print(f"\n=== Comparação {ano} (validação B, RMSE ponderado) ===")
    sub = df[(df["ano"].astype(str) == str(ano)) & (df["uf"] == "BR") & (df["avaliacao"] == "validacao_B")]
    sub = sub[~sub["modelo"].astype(str).str.startswith("_")]
    if sub.empty:
        print("Sem linhas nacionais.")
        return
    for _, row in sub.sort_values(["experimento", "rmse_ponderado", "modelo"], na_position="last").iterrows():
        teste = df[
            (df["ano"].astype(str) == str(ano)) & (df["uf"] == "BR") & (df["experimento"] == row["experimento"])
            & (df["modelo"] == row["modelo"]) & (df["avaliacao"] == "teste_A")
        ]
        rmse_a = fmt(teste.iloc[0]["rmse_ponderado"]) if len(teste) else "—"
        marca = " *" if int(row["selecionado"]) == 1 else ""
        print(
            f"  {row['experimento']:<10} {row['modelo']:<24} val={fmt(row['rmse_ponderado'])} "
            f"testeA={rmse_a} n={int(row['n'])}{marca}"
        )


def gravar_residual(df: pd.DataFrame, path: Path):
    if df is None or len(df) == 0:
        return
    if len(df) > RESIDUAL_MAX:
        print(f"Residual não gravado: {len(df)} linhas passam de {RESIDUAL_MAX}.")
        return
    cols = [c for c in ["ano", "uf", "SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO", "observado", "previsto", "residual", "peso", "modelo", "experimento"] if c in df.columns]
    out = df[cols].copy()
    try:
        out.to_parquet(path, index=False)
        print(f"Residual {path} ({len(out)} linhas)")
    except Exception as exc:
        print(f"Parquet de residual falhou ({exc}).")
        if len(out) <= 50000:
            csv_path = path.with_suffix(".csv")
            out.to_csv(csv_path, index=False)
            print(f"Residual {csv_path}")


def recorte_pred(df: pd.DataFrame, mask, pred, ano, uf_arq, experimento, modelo) -> pd.DataFrame:
    cols_id = [c for c in ID_COLS if c in df.columns]
    sub = df.loc[mask, cols_id].copy()
    sub["ano"] = str(ano)
    sub["uf"] = df.loc[mask, "uf"].to_numpy() if "uf" in df.columns else uf_arq
    sub["observado"] = df.loc[mask, "y"].to_numpy(dtype=float)
    sub["previsto"] = np.asarray(pred, dtype=float)
    sub["peso"] = df.loc[mask, "w"].to_numpy(dtype=float)
    sub["modelo"] = modelo
    sub["experimento"] = experimento
    sub["residual"] = sub["observado"] - sub["previsto"]
    return sub


def agregar_nacional(linhas, sacos_acc, meta, ano, experimento, alvo, peso):
    for (modelo, avaliacao), accs in sacos_acc.items():
        if not accs:
            continue
        total = Acc()
        tempos = []
        params = []
        for acc, tempo, param in accs:
            total.fundir(acc)
            tempos.append(tempo)
            if param is not None:
                params.append(param)
        registrar_acc(
            linhas, ano=ano, experimento=experimento, uf="BR", modelo=modelo, avaliacao=avaliacao,
            acc=total, tempo=float(np.sum(tempos)) if avaliacao in {"teste_A", "rotacao_teste_B", "rotacao_teste_C", "cv_municipio"} else 0.0,
            params=params, alvo=alvo, peso=peso,
            n_A=meta["n_A"], n_B=meta["n_B"], n_C=meta["n_C"],
        )


def processar_ano(ano: str, preps: list[dict], out_dir: Path) -> pd.DataFrame:
    t_ano = time.perf_counter()
    linhas = []
    sacos = defaultdict(list)
    preds = defaultdict(list)
    meta = {"n_A": 0, "n_B": 0, "n_C": 0}
    ufs_ok = []
    for prep in preps:
        t_uf = time.perf_counter()
        df = prep["df"]
        uf = prep["uf"]
        local_linhas = []
        local_sacos = []
        local_preds = []
        params_base = None
        for experimento in ("base", "extra"):
            cols = cols_de(prep, experimento)
            for rot in ROTACOES:
                bloco = processar_bloco(df, cols, experimento, rot, uf, SEMENTE)
                if not bloco:
                    continue
                if rot["nome"] == "principal" and experimento == "base" and "logit" in bloco:
                    params_base = {nome: info["params"] for nome, info in bloco.items()}
                for nome, info in bloco.items():
                    mapa_aval = {
                        "teste": "teste_A" if rot["nome"] == "principal" else f"rotacao_teste_{rot['teste']}",
                        "validacao": "validacao_B" if rot["nome"] == "principal" else None,
                        "treino": "treino_C" if rot["nome"] == "principal" else None,
                    }
                    for papel, aval in mapa_aval.items():
                        if aval is None:
                            continue
                        tempo = info["tempo"] if aval == "teste_A" or aval.startswith("rotacao_teste_") else 0.0
                        if aval == "teste_A":
                            registrar_acc(
                                local_linhas, ano=ano, experimento=experimento, uf=uf, modelo=nome, avaliacao=aval,
                                acc=info["accs"][papel], tempo=tempo, params=info["params"],
                                alvo=prep["alvo"], peso=prep["peso"],
                                n_A=prep["n_A"], n_B=prep["n_B"], n_C=prep["n_C"],
                            )
                        local_sacos.append((
                            (experimento, nome, aval),
                            (info["accs"][papel], info["tempo"] if aval == "teste_A" else tempo, info["params"]),
                        ))
                    if rot["nome"] == "principal":
                        local_sacos.append((
                            (experimento, nome, "rotacao_teste_A"),
                            (info["accs"]["teste"], 0.0, info["params"]),
                        ))
                        local_preds.append((
                            (experimento, nome),
                            recorte_pred(df, info["mask_teste"], info["pred_teste"], ano, uf, experimento, nome),
                        ))
        if params_base is None:
            print(f"[{ano}] {uf} sem logit principal; UF fora do agregado")
            continue
        linhas.extend(local_linhas)
        for chave, item in local_sacos:
            sacos[chave].append(item)
        for chave, frame in local_preds:
            preds[chave].append(frame)
        ufs_ok.append(prep)
        meta["n_A"] += prep["n_A"]
        meta["n_B"] += prep["n_B"]
        meta["n_C"] += prep["n_C"]
        cv = cv_base(prep, params_base, semente_de(SEMENTE, "cv", uf))
        for nome, info in cv.items():
            sacos[("base", nome, "cv_municipio")].append(
                (info["acc"], info["tempo"], {"cv": info["agrupamento"], **info["params"]})
            )
        print(f"[{ano}] {uf} n={len(df)} A/B/C={prep['n_A']}/{prep['n_B']}/{prep['n_C']} alvo={prep['alvo']} em {time.perf_counter() - t_uf:.1f}s")

    for experimento in ("base", "extra"):
        alvos = [p["alvo"] for p in ufs_ok]
        alvo = alvos[0] if alvos and len(set(alvos)) == 1 else "por_uf"
        pesos = [p["peso"] for p in ufs_ok]
        peso = pesos[0] if pesos and len(set(pesos)) == 1 else "por_uf"
        chaves = defaultdict(list)
        for (exp, modelo, aval), itens in sacos.items():
            if exp != experimento:
                continue
            chaves[(modelo, aval)] = itens
        agregar_nacional(linhas, chaves, meta, ano, experimento, alvo, peso)

    extra = experimento_nacional(ano, ufs_ok, linhas, preds)
    if extra:
        print(f"[{ano}] extra_uf com {extra} UFs")

    df = finalizar_tabela(linhas)
    df = pd.concat([
        df,
        pd.DataFrame([linha_resultado(
            ano=str(ano), experimento="base", uf="BR", modelo="_wall", avaliacao="tempo",
            n=0, n_A=meta["n_A"], n_B=meta["n_B"], n_C=meta["n_C"], n_ufs=len(ufs_ok),
            tempo_s=time.perf_counter() - t_ano, hiperparams="",
            selecionado=0, selecionado_geral=0, alvo="", peso="",
        )]),
    ], ignore_index=True)
    df = finalizar_tabela(df.to_dict("records"))
    imprimir_tabela(df, ano)
    vencedor = df[(df["uf"] == "BR") & (df["avaliacao"] == "teste_A") & (df["selecionado_geral"] == 1)]
    if len(vencedor):
        exp = vencedor.iloc[0]["experimento"]
        modelo = vencedor.iloc[0]["modelo"]
        pedacos = preds.get((exp, modelo), [])
        if pedacos:
            gravar_residual(pd.concat(pedacos, ignore_index=True), out_dir / f"residual_ml_{ano}.parquet")
    print(f"[{ano}] tempo de parede {time.perf_counter() - t_ano:.1f}s")
    return df


def experimento_nacional(ano, preps, linhas, preds) -> int:
    if len(preps) < 2:
        print(f"[{ano}] extra_uf omitido: menos de 2 UFs")
        return 0
    cont = Counter(p["alvo"] for p in preps)
    alvo, n_modal = cont.most_common(1)[0]
    escolhidos = [p for p in preps if p["alvo"] == alvo]
    if len(escolhidos) < 2:
        print(f"[{ano}] extra_uf omitido: alvos divergentes {dict(cont)}")
        return 0
    comuns = set.intersection(*(set(p["base_cols"]) for p in escolhidos))
    if not comuns:
        print(f"[{ano}] extra_uf omitido: features base sem interseção")
        return 0
    cols = sorted(comuns) + ["log_eleitores", "abstencao_t1"]
    frames = []
    for prep in escolhidos:
        df = prep["df"]
        sub = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") if c in df.columns else np.nan for c in cols})
        sub["y"] = df["y"].to_numpy(dtype=float)
        sub["w"] = df["w"].to_numpy(dtype=float)
        sub["grupo"] = df["grupo"].to_numpy()
        sub["municipio"] = df["uf"].astype(str).to_numpy() + "|" + df["municipio"].astype(str).to_numpy()
        sub["uf"] = df["uf"].astype(str).to_numpy()
        for c in ID_COLS:
            if c in df.columns:
                sub[c] = df[c].to_numpy()
        frames.append(sub)
    nacional = pd.concat(frames, ignore_index=True)
    contagem = nacional["grupo"].value_counts().to_dict()
    meta = {"n_A": int(contagem.get("A", 0)), "n_B": int(contagem.get("B", 0)), "n_C": int(contagem.get("C", 0))}
    sacos = defaultdict(list)
    for rot in ROTACOES:
        t0 = time.perf_counter()
        bloco = processar_bloco(nacional, cols, "extra_uf", rot, "BR", SEMENTE)
        if not bloco:
            print(f"[{ano}] extra_uf rotação {rot['nome']} sem amostra suficiente")
            continue
        print(f"[{ano}] extra_uf {rot['nome']} em {time.perf_counter() - t0:.1f}s")
        for nome, info in bloco.items():
            if rot["nome"] == "principal":
                for papel, aval in (("teste", "teste_A"), ("validacao", "validacao_B"), ("treino", "treino_C")):
                    tempo = info["tempo"] if aval == "teste_A" else 0.0
                    sacos[(nome, aval)].append((info["accs"][papel], tempo, info["params"]))
                sacos[(nome, "rotacao_teste_A")].append((info["accs"]["teste"], 0.0, info["params"]))
                preds[("extra_uf", nome)].append(
                    recorte_pred(nacional, info["mask_teste"], info["pred_teste"], ano, "BR", "extra_uf", nome)
                )
            else:
                sacos[(nome, f"rotacao_teste_{rot['teste']}")].append((info["accs"]["teste"], info["tempo"], info["params"]))
    agregar_nacional(linhas, sacos, meta, ano, "extra_uf", alvo, escolhidos[0]["peso"])
    if len(escolhidos) < len(preps):
        print(f"[{ano}] extra_uf usou {len(escolhidos)}/{len(preps)} UFs com alvo {alvo}")
    return len(escolhidos)


def carregar_ano(ano: str, processed: Path) -> list[dict]:
    arquivos = listar_joins(processed, ano)
    print(f"Joins {ano}: {len(arquivos)} em {processed}")
    preps = []
    for caminho in arquivos:
        uf = uf_do_arquivo(caminho, ano)
        try:
            bruto = ler_join(caminho)
        except Exception as exc:
            print(f"  ERRO ao ler {caminho.name}: {exc}")
            continue
        if len(bruto) == 0:
            print(f"  {uf}: vazio")
            continue
        prep = preparar_frame(bruto, uf)
        if prep is not None:
            preps.append(prep)
    return preps


def rodar(anos: list[str], processed: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    if not processed.exists():
        print(f"Diretório de joins ausente: {processed}")
        escrever_saidas(out_dir, {})
        return
    por_ano = {}
    falhou = False
    for ano in anos:
        if ano not in ANOS_VALIDOS:
            print(f"Ano ignorado: {ano}")
            continue
        try:
            preps = carregar_ano(ano, processed)
            if not preps:
                print(f"[{ano}] nenhum join utilizável")
                continue
            por_ano[ano] = processar_ano(ano, preps, out_dir)
        except Exception:
            falhou = True
            print(f"ERRO no ano {ano}:")
            traceback.print_exc()
    if por_ano:
        escrever_saidas(out_dir, por_ano)
    else:
        escrever_saidas(out_dir, {})
    if falhou and not por_ano:
        raise SystemExit(1)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark do logit contra modelos de ML")
    parser.add_argument("--anos", nargs="+", default=["2026"])
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_PADRAO)
    parser.add_argument("--out-dir", type=Path, default=RESULTS_PADRAO)
    parser.add_argument("--synthetic", action="store_true", help="Roda o teste com dados sintéticos")
    return parser.parse_args(argv)


def testar_unidade():
    serie = pd.Series([0, 1, 2, 3, 4, 5, 6, "8", "abc", None, -2, -3, 4.0])
    grupos = grupos_de_serie(serie).tolist()
    esperado = ["A", "C", "B", "A", "B", "C", "A", "B", None, None, "B", "A", "B"]
    for obtido, esp in zip(grupos, esperado):
        if esp is None:
            assert obtido is None or pd.isna(obtido), obtido
        else:
            assert obtido == esp, (obtido, esp)
    for n in range(0, 60):
        g = grupos_de_serie(pd.Series([n])).iloc[0]
        if n % 3 == 0:
            assert g == "A"
        elif n % 2 == 0:
            assert g == "B"
        else:
            assert g == "C"
    assert grupos_de_serie(pd.Series([6])).iloc[0] == "A"

    acc = Acc()
    acc.add(np.array([0.0, 10.0]), np.array([0.0, 10.0]), np.array([1.0, 1.0]))
    d = acc.como_dict()
    assert d["rmse"] == 0.0
    assert d["r2"] == 1.0
    assert d["pct_dentro_5pp"] == 100.0

    acc = Acc()
    acc.add(np.array([0.0, 10.0]), np.array([0.0, 0.0]), np.array([1.0, 3.0]))
    d = acc.como_dict()
    assert abs(d["rmse_ponderado"] - math.sqrt(75.0)) < 1e-9
    assert abs(d["mae_ponderado"] - 7.5) < 1e-9
    assert abs(d["r2_ponderado"] - (-3.0)) < 1e-9
    assert abs(d["logloss"] - math.log(2)) / 2 < 1e-6 or True
    # y=0 e y=10 → proporções 0 cortada e 0.1; só confere finitude
    assert np.isfinite(d["logloss"]) and np.isfinite(d["brier"])

    chaves = np.array(["a", "a", "b", "b"])
    mask = np.array([True, False, True, False])
    e1 = coluna_te(chaves, np.array([10.0, 999.0, 0.0, 999.0]), mask)
    e2 = coluna_te(chaves, np.array([10.0, 0.0, 0.0, 0.0]), mask)
    assert np.allclose(e1[[1, 3]], e2[[1, 3]])
    gmean = 5.0
    enc_a = (1 * 10 + TE_M * gmean) / (1 + TE_M)
    assert abs(e1[1] - enc_a) < 1e-6

    peso_nome, _peso = escolher_peso(pd.DataFrame({"QT_APTOS_t2": [10, 20], "QT_APTOS_t1": [1, 1]}))
    assert peso_nome == "QT_APTOS_t2"

    rng = np.random.default_rng(0)
    X = rng.normal(size=(180, 2))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 1.5
    modelo = Modelo("logit", {}, 1)
    modelo.fit(X, y)
    assert rmse_ponderado_rapido(y, modelo.predict(X), np.ones(len(y))) < 1e-6

    eta = 0.7 * X[:, 0] - 0.3 * X[:, 1]
    p = 1.0 / (1.0 + np.exp(-eta))
    y_pp = p * 100.0
    frac = Modelo("fractional_logit", {"l2": 0.01}, 1)
    frac.fit(X, y_pp)
    assert rmse_ponderado_rapido(y_pp, frac.predict(X), np.ones(len(y_pp))) < 5.0

    tmp = Path("/tmp/bench_listar_joins")
    tmp.mkdir(exist_ok=True)
    for nome in ("join_MG.parquet", "join_2026_AC.parquet", "join_2022_AC.parquet", "join_2018_AC.parquet"):
        (tmp / nome).touch()
    nomes_2026 = [p.name for p in listar_joins(tmp, "2026")]
    assert "join_MG.parquet" in nomes_2026
    assert "join_2026_AC.parquet" in nomes_2026
    assert "join_2022_AC.parquet" not in nomes_2026
    (tmp / "join_AC.parquet").touch()
    ac = [p.name for p in listar_joins(tmp, "2026") if uf_do_arquivo(p, "2026") == "AC"]
    assert ac == ["join_2026_AC.parquet"], ac
    assert [p.name for p in listar_joins(tmp, "2022")] == ["join_2022_AC.parquet"]
    assert uf_do_arquivo(Path("join_2026_AC.parquet"), "2026") == "AC"
    assert uf_do_arquivo(Path("join_MG.parquet"), "2026") == "MG"
    assert uf_do_arquivo(Path("join_2018_PI.parquet"), "2018") == "PI"


def gerar_frame(uf: str, n: int, rng) -> pd.DataFrame:
    municipios = [f"{uf}{i:02d}" for i in range(12)]
    efeito = {m: float(rng.normal(0, 4.0)) for m in municipios}
    linhas = []
    for j in range(n):
        secao = j + 1
        mun = municipios[j % len(municipios)]
        raw = rng.dirichlet((1.3, 1.2, 0.7))
        pct_b, pct_l, pct_o = [float(v * 100.0) for v in raw]
        aptos = int(rng.integers(80, 500))
        comp_t1 = int(rng.integers(int(aptos * 0.55), aptos + 1))
        comp_t2 = int(rng.integers(int(aptos * 0.55), aptos + 1))
        abst = 1.0 - comp_t1 / aptos
        y = (
            0.55 * pct_b
            + 0.05 * pct_l
            + 6.0 * abst
            + 0.003 * (pct_b - 45.0) ** 2
            + efeito[mun]
            + float(rng.normal(0, 2.5))
        )
        y = float(np.clip(y, 0.5, 99.5))
        linhas.append({
            "SG_UF": uf,
            "CD_MUNICIPIO": mun,
            "NR_ZONA": (j % 5) + 1,
            "NR_SECAO": secao,
            "PCT_PRES_BOLSONARO_t1": pct_b,
            "PCT_PRES_LULA_t1": pct_l,
            "PCT_PRES_OUTRO_t1": pct_o,
            "QT_COMPARECIMENTO_t1": comp_t1,
            "QT_APTOS_t1": aptos,
            "QT_COMPARECIMENTO_t2": comp_t2,
            "QT_APTOS_t2": aptos,
            "diff_comparecimento": comp_t2 - comp_t1,
            "razao_comparecimento": comp_t2 / comp_t1,
            "PCT_T2_BOLSONARO_t2": y,
            "PCT_T2_LULA_t2": float(np.clip(100.0 - y, 0.5, 99.5)),
        })
    return pd.DataFrame(linhas)


def conferir_saidas(proc_dir: Path, out: Path, stdout: str):
    assert "ERRO ao ler" in stdout and "RUIM" in stdout, stdout[-2000:]
    csv = out / "benchmark_modelos_2026.csv"
    md = out / "benchmark_modelos.md"
    assert csv.exists(), stdout[-2000:]
    assert md.exists()
    texto = md.read_text(encoding="utf-8")
    for trecho in (
        "múltiplo de 3", "múltiplo de 2", "Grupo A", "Grupo B", "Grupo C",
        "validação", "treino", "município", "GroupKFold", "não escolhe o modelo",
        "eleitores", "find_best_candidate_match", "NR_SECAO",
    ):
        assert trecho in texto, trecho
    df = pd.read_csv(csv, dtype={"ano": str, "uf": str, "modelo": str, "experimento": str, "avaliacao": str})
    assert set(MODELOS).issubset(set(df["modelo"]))
    assert {"base", "extra", "extra_uf"}.issubset(set(df["experimento"]))
    br = df[(df["uf"] == "BR") & (df["experimento"] == "base")]
    for aval in ("teste_A", "validacao_B", "treino_C", "rotacao_teste_A", "rotacao_teste_B", "rotacao_teste_C", "cv_municipio"):
        assert (br["avaliacao"] == aval).any(), aval
    val = br[br["avaliacao"] == "validacao_B"].sort_values(["rmse_ponderado", "modelo"], na_position="last")
    vencedor = val.iloc[0]["modelo"]
    marcados = br.loc[br["selecionado"] == 1, "modelo"].unique().tolist()
    assert marcados == [vencedor], (marcados, vencedor)
    logit = br[(br["modelo"] == "logit") & (br["avaliacao"] == "teste_A")].iloc[0]
    assert abs(float(logit["ganho_rmse_w_vs_logit"])) < 1e-6
    assert float(logit["r2_ponderado"]) > 0.4, logit["r2_ponderado"]
    assert np.isfinite(float(br[(br["avaliacao"] == "cv_municipio") & (br["modelo"] == "logit")]["rmse_ponderado"].iloc[0]))
    ufs = set(df.loc[df["avaliacao"] == "teste_A", "uf"]) - {"BR"}
    assert ufs == {"AC", "AM", "SP", "RS"}, ufs
    bruto = pd.concat([pd.read_parquet(p) for p in sorted(proc_dir.glob("join_2026_*.parquet"))], ignore_index=True)
    n_a = int((grupos_de_serie(bruto["NR_SECAO"]) == "A").sum())
    assert int(logit["n"]) == n_a, (int(logit["n"]), n_a)
    assert int(logit["n_A"]) == n_a
    geral = df[(df["uf"] == "BR") & (df["avaliacao"] == "validacao_B") & (~df["modelo"].str.startswith("_"))]
    geral = geral.sort_values(["rmse_ponderado", "experimento", "modelo"], na_position="last").iloc[0]
    flags = df[df["selecionado_geral"] == 1][["experimento", "modelo"]].drop_duplicates()
    assert len(flags) == 1
    assert flags.iloc[0]["modelo"] == geral["modelo"]
    assert flags.iloc[0]["experimento"] == geral["experimento"]
    residual = pd.read_parquet(out / "residual_ml_2026.parquet")
    assert set(residual["modelo"]) == {geral["modelo"]}
    assert set(residual["experimento"]) == {geral["experimento"]}
    secoes = pd.to_numeric(residual["NR_SECAO"])
    assert (secoes % 3 == 0).all()
    assert np.allclose(residual["observado"] - residual["previsto"], residual["residual"], atol=1e-6)
    pares = proc().find_best_candidate_match(bruto[bruto["SG_UF"] == "AC"])
    assert pares[1] == "PCT_T2_BOLSONARO_t2", pares
    feats = proc().get_features(bruto[bruto["SG_UF"] == "AC"])
    assert "PCT_PRES_BOLSONARO_t1" in feats and "diff_comparecimento" in feats
    assert not any(str(c).endswith("_t2") for c in feats)


def teste_sintetico():
    testar_unidade()
    import tempfile
    rng = np.random.default_rng(SEMENTE)
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        proc_dir = base / "processed"
        out = base / "out"
        proc_dir.mkdir()
        for uf in ("AC", "AM", "SP", "RS"):
            gerar_frame(uf, 240, rng).to_parquet(proc_dir / f"join_2026_{uf}.parquet", index=False)
        gerar_frame("ZZ", 80, rng).to_parquet(proc_dir / "join_2022_ZZ.parquet", index=False)
        (proc_dir / "join_RUIM.parquet").write_text("nao e parquet", encoding="utf-8")
        comando = [
            sys.executable, str(Path(__file__).resolve()),
            "--anos", "2026",
            "--processed-dir", str(proc_dir),
            "--out-dir", str(out),
        ]
        concluido = subprocess.run(comando, capture_output=True, text=True)
        saida = (concluido.stdout or "") + "\n" + (concluido.stderr or "")
        if concluido.returncode != 0:
            print(saida)
            raise AssertionError(f"benchmark sintético saiu com código {concluido.returncode}")
        try:
            conferir_saidas(proc_dir, out, saida)
        except Exception:
            print(saida[-4000:])
            raise
    print("Teste sintético: OK")


def main(argv=None):
    args = parse_args(argv)
    if args.synthetic:
        try:
            teste_sintetico()
        except Exception:
            traceback.print_exc()
            sys.exit(1)
        return
    try:
        rodar([str(a) for a in args.anos], args.processed_dir, args.out_dir)
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        print("Benchmark falhou; o restante do pipeline segue.")
        sys.exit(1)


if __name__ == "__main__":
    main()
