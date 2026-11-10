#!/usr/bin/env python3
"""
Cruza o residual de seção (T1→T2) agregado por município com IDHM, transferências
e Bolsa Família.

O residual não está nos joins: é o mesmo de scripts/process_and_summarize.py
(residual_distribuicao — regressão linear, observado − previsto, em pontos
percentuais, nas seções de teste NR_SECAO % 3 != 0). A média municipal é
ponderada por QT_COMPARECIMENTO_t2 (senão QT_APTOS_t2, senão peso 1).

Bolsa Família vem com vários mes_referencia por município. Fica o mês mais
recente (não a média), para não misturar competências nem contar o mesmo
beneficiário mais de uma vez.

Crosswalk TSE ↔ IBGE ↔ SIAFI, vendido em data/referencia/municipios_tse_ibge_siafi.csv:
- codigo_tse e codigo_ibge: betafcc/Municipios-Brasileiros-TSE
- codigo_siafi: kelvins/municipios-brasileiros (siafi_id), junção por codigo_ibge
Nome+UF normalizado só cobre o que o código não achou.

Uso:
  python scripts/cruzar_socioeconomico.py --ano 2026
"""

from __future__ import annotations

from pathlib import Path
import functools
import re
import sys
import unicodedata

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass
print = functools.partial(print, flush=True)

if __package__:
    from .process_and_summarize import (
        find_best_candidate_match,
        get_features,
        residual_distribuicao,
        split_mult3,
    )
    from . import baixar_socioeconomico as gastos
else:
    from process_and_summarize import (
        find_best_candidate_match,
        get_features,
        residual_distribuicao,
        split_mult3,
    )
    import baixar_socioeconomico as gastos

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = Path("/home/alexa/eleicoes-data/processed")
DATA_SOCIO = ROOT / "data" / "socioeconomico"
DATA_REF = ROOT / "data" / "referencia" / "municipios_tse_ibge_siafi.csv"
RESULTS = ROOT / "resultados"

URL_TSE_IBGE = "https://raw.githubusercontent.com/betafcc/Municipios-Brasileiros-TSE/master/municipios_brasileiros_tse.csv"
URL_SIAFI = "https://raw.githubusercontent.com/kelvins/municipios-brasileiros/main/csv/municipios.csv"

_UF_POR_CODIGO_IBGE = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
    31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS",
    50: "MS", 51: "MT", 52: "GO", 53: "DF",
}


def _sem_acento(texto: str) -> str:
    base = unicodedata.normalize("NFKD", str(texto))
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    base = base.lower().replace("_", " ").replace("-", " ")
    return " ".join(base.split())


# Grafia do Tesouro (sem apóstrofo: "DOeste", "DAliança", "Major Izidoro") → IBGE de 7 dígitos.
# A chave é (UF, norm_nome_chave). Municípios só de um lado (Fernando de Noronha, Boa Esperança do Norte) ficam de fora.
_ALIAS_TESOURO_IBGE = (
    ("AL", "Major Izidoro", "2704401"),
    ("AP", "Pedra Branca do Amapari", "1600154"),
    ("BA", "Muquem de Sao Francisco", "2922250"),
    ("BA", "Quijingue", "2925907"),
    ("BA", "Santa Teresinha", "2928505"),
    ("MG", "Brasopolis", "3108909"),
    ("MG", "Dona Euzebia", "3122900"),
    ("MG", "Ponte Chique", "3152131"),
    ("MT", "Poxoreo", "5107008"),
    ("PB", "Joca Claudino", "2513653"),
    ("PB", "Sao Domingos", "2513968"),
    ("PE", "Lagoa do Itaenga", "2608503"),
    ("PI", "Capitao Gervasio de Oliveira", "2202455"),
    ("PI", "Pau DArco", "2207793"),
    ("RN", "Acu", "2400208"),
    ("RN", "Januario Cicco", "2405306"),
    ("RN", "Presidente Juscelino", "2410306"),
    ("SC", "Luiz Alves", "4210001"),
    ("SE", "Gracho Cardoso", "2802601"),
    ("SP", "Florinia", "3516101"),
    ("TO", "Sao Valerio", "1720499"),
    ("TO", "Tabocao", "1708254"),
)

UFS_BRASILEIRAS = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO",
}


def norm_nome(valor) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    texto = str(valor).strip()
    if texto.lower() in {"", "nan", "none", "<na>"}:
        return ""
    texto = unicodedata.normalize("NFKD", texto)
    texto = "".join(ch for ch in texto if not unicodedata.combining(ch))
    texto = texto.upper().replace("'", " ").replace("’", " ").replace("`", " ")
    texto = texto.replace("-", " ")
    texto = re.sub(r"[^A-Z0-9 ]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def _digitos(valor) -> str:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ""
    texto = str(valor).strip()
    if texto.lower() in {"", "nan", "none", "<na>"}:
        return ""
    if re.fullmatch(r"\d+\.0", texto):
        texto = texto[:-2]
    return re.sub(r"\D", "", texto)


def norm_tse(valor) -> str:
    digitos = _digitos(valor)
    if not digitos:
        return ""
    return str(int(digitos))


def norm_siafi(valor) -> str:
    return norm_tse(valor)


def norm_ibge(valor) -> str:
    digitos = _digitos(valor)
    if not digitos:
        return ""
    if len(digitos) == 6:
        return digitos
    return digitos.zfill(7)


def norm_nome_chave(valor) -> str:
    """Une D'Oeste/DOeste/Do Oeste e D'Água/DAgua para casar Tesouro e TSE."""
    nome = norm_nome(valor)
    nome = re.sub(r"\bD\s+", "D", nome)
    nome = re.sub(r"\bDO\s+OESTE\b", "DOESTE", nome)
    return nome


def norm_uf(valor) -> str:
    texto = str(valor).strip() if valor is not None else ""
    if texto.lower() in {"", "nan", "none", "<na>"}:
        return ""
    if re.fullmatch(r"[A-Za-z]{2}", texto):
        return texto.upper()
    digitos = _digitos(texto)
    if digitos:
        try:
            return _UF_POR_CODIGO_IBGE.get(int(digitos), texto.upper())
        except ValueError:
            return texto.upper()
    return texto.upper()


def _na_se_vazio(serie: pd.Series) -> pd.Series:
    texto = serie.astype("string").str.strip()
    return texto.mask(texto.isna() | texto.eq("") | texto.str.lower().isin(["nan", "none", "<na>"]))


def _coluna(df: pd.DataFrame, *opcoes: str) -> str | None:
    indice = {}
    for coluna in df.columns:
        chave = _sem_acento(str(coluna))
        indice.setdefault(chave, coluna)
        indice.setdefault(chave.replace(" ", ""), coluna)
    for opcao in opcoes:
        chave = _sem_acento(opcao)
        if chave in indice:
            return indice[chave]
        compacta = chave.replace(" ", "")
        if compacta in indice:
            return indice[compacta]
    return None


def _numero_um(valor):
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return np.nan
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        return float(valor)
    texto = str(valor).strip()
    if texto.lower() in {"", "nan", "none", "<na>", "nat"}:
        return np.nan
    if "," in texto:
        texto = texto.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", texto) and (texto.count(".") > 1 or texto.endswith(".000")):
        texto = texto.replace(".", "")
    try:
        return float(texto)
    except ValueError:
        return np.nan


def _numero(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")
    return serie.map(_numero_um)


def _ler_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", dtype=str, comment="#", encoding="utf-8")


def listar_joins(ano: str) -> list[Path]:
    if not PROCESSED.exists():
        return []
    if str(ano) == "2026":
        return [
            p for p in sorted(PROCESSED.glob("join_*.parquet"))
            if not any(marca in p.stem for marca in ("_2018_", "_2022_"))
        ]
    return sorted(PROCESSED.glob(f"join_{ano}_*.parquet"))


def calcular_residuais_secao(ano: str) -> pd.DataFrame:
    """Recalcula o residual de seção com a mesma função do resumo por ano."""
    joins = listar_joins(ano)
    print(f"Joins para {ano}: {len(joins)}")
    if not joins:
        print("Nenhum join encontrado. Rode o preprocess primeiro.")
        return pd.DataFrame()

    frames = []
    for caminho in joins:
        uf_arquivo = caminho.stem.replace(f"join_{ano}_", "").replace("join_", "")
        try:
            merged = pd.read_parquet(caminho)
        except Exception as exc:
            print(f"  Erro ao ler {caminho.name}: {exc}")
            continue
        if merged.empty:
            continue
        merged = merged.reset_index(drop=True)
        col_t1, col_t2 = find_best_candidate_match(merged)
        if not col_t1 or not col_t2:
            print(f"  {caminho.name}: sem par de candidatos T1/T2")
            continue
        train, test = split_mult3(merged)
        if train is None or len(train) < 30:
            n_treino = 0 if train is None else len(train)
            print(f"  {caminho.name}: treino insuficiente ({n_treino})")
            continue
        features = get_features(merged)
        train_clean = train.dropna(subset=[col_t2, *features])
        test_clean = test.dropna(subset=[col_t2, *features])
        if len(train_clean) < 30:
            print(f"  {caminho.name}: treino limpo insuficiente ({len(train_clean)})")
            continue
        _info, _modelo, res_df = residual_distribuicao(train_clean, test_clean, features, col_t2, uf_arquivo)
        if res_df is None or res_df.empty or "CD_MUNICIPIO" not in res_df.columns:
            print(f"  {caminho.name}: residual sem CD_MUNICIPIO")
            continue
        if "SG_UF" not in res_df.columns:
            res_df["SG_UF"] = uf_arquivo
        nome_col = _coluna(test_clean, "nm_municipio", "nome_municipio", "municipio")
        if nome_col:
            res_df["NM_MUNICIPIO"] = test_clean.loc[res_df.index, nome_col].values
        res_df["modelo_t1"] = col_t1
        res_df["modelo_t2"] = col_t2
        frames.append(res_df)
        print(f"  {caminho.name}: {len(res_df)} seções de teste, par {col_t1} → {col_t2}")

    if not frames:
        print("Não foi possível agregar residual por município. Verifique colunas nos joins.")
        return pd.DataFrame()
    secoes = pd.concat(frames, ignore_index=True)
    chaves = [c for c in ("SG_UF", "CD_MUNICIPIO", "NR_ZONA", "NR_SECAO") if c in secoes.columns]
    if len(chaves) >= 3:
        antes = len(secoes)
        secoes = secoes.drop_duplicates(chaves)
        if len(secoes) < antes:
            print(f"Seções duplicadas removidas: {antes - len(secoes)}")
    return secoes


def agregar_residual_municipio(secoes: pd.DataFrame) -> pd.DataFrame:
    df = secoes.copy()
    if "SG_UF" in df.columns:
        df["sg_uf"] = df["SG_UF"].map(norm_uf)
    else:
        df["sg_uf"] = df["uf"].map(norm_uf) if "uf" in df.columns else ""
    df["cd_municipio_tse"] = df["CD_MUNICIPIO"].map(norm_tse)
    if "NM_MUNICIPIO" in df.columns:
        df["nome_bruto"] = df["NM_MUNICIPIO"]
    else:
        df["nome_bruto"] = pd.NA
    comparecimento = _numero(df["QT_COMPARECIMENTO_t2"]) if "QT_COMPARECIMENTO_t2" in df.columns else pd.Series(0.0, index=df.index)
    eleitores = _numero(df["QT_APTOS_t2"]) if "QT_APTOS_t2" in df.columns else pd.Series(0.0, index=df.index)
    comparecimento = comparecimento.fillna(0)
    eleitores = eleitores.fillna(0)
    # Peso da seção: comparecimento; se zerado, eleitores; se ainda zerado, 1.
    peso = comparecimento.where(comparecimento > 0, eleitores)
    peso = peso.where(peso > 0, 1.0)
    df["peso"] = peso
    df["eleitores_secao"] = eleitores
    df["comparecimento_secao"] = comparecimento
    df["residual_x_peso"] = _numero(df["residual"]) * peso
    chaves = ["sg_uf", "cd_municipio_tse"]
    agrupado = df.groupby(chaves, dropna=False)
    saida = agrupado.agg(
        n_secoes=("residual", "size"),
        n_eleitores=("eleitores_secao", "sum"),
        n_comparecimento=("comparecimento_secao", "sum"),
        residual_mean_unweighted=("residual", "mean"),
        residual_std=("residual", "std"),
        soma_peso=("peso", "sum"),
        soma_residual_peso=("residual_x_peso", "sum"),
    ).reset_index()
    nomes = (
        df.dropna(subset=["nome_bruto"])
        .assign(nome_bruto=lambda t: t["nome_bruto"].astype(str).str.strip())
    )
    nomes = nomes[nomes["nome_bruto"].ne("") & nomes["nome_bruto"].str.lower().ne("nan")]
    if not nomes.empty:
        nome_por_muni = nomes.groupby(chaves)["nome_bruto"].agg(lambda s: s.iloc[0])
        saida = saida.merge(nome_por_muni.rename("nome_municipio"), on=chaves, how="left")
    else:
        saida["nome_municipio"] = pd.NA
    saida["residual_mean"] = saida["soma_residual_peso"] / saida["soma_peso"].replace(0, np.nan)
    saida["n_eleitores"] = saida["n_eleitores"].round().astype("int64")
    saida["n_comparecimento"] = saida["n_comparecimento"].round().astype("int64")
    saida["n_secoes"] = saida["n_secoes"].astype("int64")
    return saida.drop(columns=["soma_peso", "soma_residual_peso"])


def _baixar_crosswalk(destino: Path) -> pd.DataFrame | None:
    try:
        import requests
    except ImportError:
        print("AVISO: requests indisponível; crosswalk não baixado")
        return None
    try:
        tse = pd.read_csv(URL_TSE_IBGE, dtype=str)
        siafi = pd.read_csv(URL_SIAFI, dtype=str)
    except Exception as exc:
        print(f"AVISO: falha ao baixar crosswalk ({exc})")
        return None
    cruzado = tse.merge(siafi[["codigo_ibge", "siafi_id"]], on="codigo_ibge", how="left")
    cruzado = cruzado.rename(columns={"siafi_id": "codigo_siafi"})
    colunas = ["codigo_tse", "codigo_ibge", "codigo_siafi", "nome_municipio", "uf", "capital"]
    cruzado = cruzado[colunas]
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", encoding="utf-8", newline="") as arquivo:
        arquivo.write(
            "# fonte: betafcc/Municipios-Brasileiros-TSE + kelvins/municipios-brasileiros; juncao por codigo_ibge\n"
        )
        cruzado.to_csv(arquivo, index=False)
    print(f"Crosswalk baixado em {destino} ({len(cruzado)} municípios)")
    return cruzado


def carregar_crosswalk() -> pd.DataFrame:
    cache = DATA_SOCIO / DATA_REF.name
    origem = DATA_REF if DATA_REF.exists() else cache if cache.exists() else None
    if origem is None:
        baixado = _baixar_crosswalk(DATA_REF)
        if baixado is None:
            print("AVISO: sem crosswalk TSE↔IBGE↔SIAFI")
            return pd.DataFrame()
        tabela = baixado
        origem = DATA_REF
    else:
        tabela = pd.read_csv(origem, dtype=str, comment="#")
        print(f"Crosswalk: {origem} ({len(tabela)} municípios)")
    try:
        DATA_SOCIO.mkdir(parents=True, exist_ok=True)
        if origem.resolve() != cache.resolve():
            cache.write_text(origem.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Crosswalk em cache: {cache}")
    except Exception as exc:
        print(f"Aviso: não copiei o crosswalk para {cache}: {exc}")
    return tabela


def anexar_crosswalk(municipios: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    saida = municipios.copy()
    saida["match_crosswalk"] = "sem_match"
    if crosswalk.empty:
        saida["codigo_ibge"] = pd.NA
        saida["codigo_siafi"] = pd.NA
        return saida
    tabela = crosswalk.copy()
    tabela["cd_municipio_tse"] = tabela["codigo_tse"].map(norm_tse)
    tabela["sg_uf"] = tabela["uf"].map(norm_uf)
    tabela["codigo_ibge"] = _na_se_vazio(tabela["codigo_ibge"].map(norm_ibge))
    tabela["codigo_siafi"] = _na_se_vazio(tabela["codigo_siafi"].map(norm_siafi))
    tabela["nome_cw"] = tabela["nome_municipio"]
    base = tabela.drop_duplicates(["cd_municipio_tse", "sg_uf"])
    saida = saida.merge(
        base[["cd_municipio_tse", "sg_uf", "codigo_ibge", "codigo_siafi", "nome_cw"]],
        on=["cd_municipio_tse", "sg_uf"],
        how="left",
    )
    casou = saida["codigo_ibge"].notna()
    saida.loc[casou, "match_crosswalk"] = "codigo"
    saida["nome_municipio"] = saida["nome_cw"].fillna(saida["nome_municipio"])
    saida = saida.drop(columns=["nome_cw"])

    faltam = saida["codigo_ibge"].isna() & saida["nome_municipio"].notna()
    if faltam.any():
        tabela["chave"] = tabela["nome_municipio"].map(norm_nome) + "|" + tabela["sg_uf"]
        unicos = tabela[tabela["chave"].ne("|")].drop_duplicates("chave", keep=False)
        mapa = unicos.set_index("chave")
        chaves = saida.loc[faltam, "nome_municipio"].map(norm_nome) + "|" + saida.loc[faltam, "sg_uf"].map(norm_uf)
        uteis = chaves[chaves.str.len().gt(1) & ~chaves.str.startswith("|") & chaves.isin(mapa.index)]
        if not uteis.empty:
            escolhidos = mapa.loc[uteis.to_numpy()]
            saida.loc[uteis.index, "codigo_ibge"] = escolhidos["codigo_ibge"].to_numpy()
            saida.loc[uteis.index, "codigo_siafi"] = escolhidos["codigo_siafi"].to_numpy()
            saida.loc[uteis.index, "nome_municipio"] = escolhidos["nome_cw"].to_numpy()
            saida.loc[uteis.index, "match_crosswalk"] = "nome"
    n = len(saida)
    por_codigo = int((saida["match_crosswalk"] == "codigo").sum())
    por_nome = int((saida["match_crosswalk"] == "nome").sum())
    print(f"Crosswalk TSE: {por_codigo}/{n} por código, {por_nome}/{n} por nome+UF, {n - por_codigo - por_nome}/{n} sem match")
    return saida


def _filtrar_ano_2010(df: pd.DataFrame) -> pd.DataFrame:
    col_ano = _coluna(df, "ano")
    if not col_ano:
        return df
    ano = _numero(df[col_ano])
    if (ano == 2010).any():
        return df.loc[ano == 2010].copy()
    if ano.notna().any():
        ultimo = ano.max()
        print(f"AVISO: IDHM sem 2010; usando ano {int(ultimo)}")
        return df.loc[ano == ultimo].copy()
    return df


def carregar_idh() -> pd.DataFrame:
    caminho = DATA_SOCIO / "idh_municipios.csv"
    if not caminho.exists():
        print(f"AVISO: {caminho} não encontrado")
        return pd.DataFrame()
    df = _filtrar_ano_2010(_ler_csv(caminho))
    col_ibge = _coluna(df, "codigo_ibge", "codmun7", "cd_mun", "codmun6")
    col_nome = _coluna(df, "municipio", "nome_municipio", "nome")
    col_uf = _coluna(df, "uf")
    saida = pd.DataFrame(index=df.index)
    if col_ibge:
        saida["codigo_ibge"] = _na_se_vazio(df[col_ibge].map(norm_ibge))
    if col_nome:
        saida["nome_idh"] = df[col_nome]
    if col_uf:
        saida["uf_idh"] = df[col_uf].map(norm_uf)
    for destino, opcoes in (
        ("idhm", ("idhm",)),
        ("idhm_e", ("idhm_e", "idhm e")),
        ("idhm_l", ("idhm_l", "idhm l")),
        ("idhm_r", ("idhm_r", "idhm r")),
        ("pop", ("pop", "populacao")),
    ):
        coluna = _coluna(df, *opcoes)
        if coluna:
            saida[destino] = _numero(df[coluna])
    if "codigo_ibge" in saida.columns:
        saida = saida.dropna(subset=["codigo_ibge"]).drop_duplicates("codigo_ibge")
    print(f"IDHM carregado: {len(saida)} municípios")
    return saida.reset_index(drop=True)


def _classificar_codigo_bruto(valor) -> tuple[str, str]:
    """Devolve (tipo, chave) com tipo em ibge, ibge6, siafi ou vazio.

    Aceita texto, inteiro e float no formato '1200401.0' / '0139.0'.
    7 dígitos = IBGE; 6 = IBGE sem dígito verificador; 1–4 = SIAFI.
    """
    digitos = _digitos(valor)
    if not digitos:
        return "", ""
    if len(digitos) > 7:
        digitos = digitos[-7:]
    if len(digitos) == 7:
        return "ibge", digitos
    if len(digitos) == 6:
        return "ibge6", digitos
    if 1 <= len(digitos) <= 4:
        return "siafi", str(int(digitos))
    return "", ""


def _coluna_codigo_municipio(df: pd.DataFrame) -> str | None:
    ignorar = ("transferencia", "decendio", "item", "valor", "ano", "mes", "populacao", "benefici")
    candidatas = []
    for coluna in df.columns:
        chave = _sem_acento(str(coluna))
        if any(parte in chave for parte in ignorar):
            continue
        if any(parte in chave for parte in ("ibge", "siafi", "codmun", "cd mun", "codigo")):
            candidatas.append(coluna)
    if not candidatas:
        return None

    def pontuar(coluna: str) -> int:
        tipos = df[coluna].map(lambda v: _classificar_codigo_bruto(v)[0])
        bonus = 10_000_000 if "ibge" in _sem_acento(str(coluna)) else 0
        return bonus + int(tipos.isin(["ibge", "ibge6", "siafi"]).sum())

    melhor = max(candidatas, key=pontuar)
    if pontuar(melhor) == 0:
        return None
    return melhor


def _mapas_crosswalk(crosswalk: pd.DataFrame) -> dict:
    if crosswalk is None or crosswalk.empty:
        return {"ibge": {}, "ibge6": {}, "siafi": {}, "nome": {}, "alias": {}}
    tabela = crosswalk.copy()
    tabela["ibge"] = _na_se_vazio(tabela["codigo_ibge"].map(norm_ibge)).fillna("").astype(str)
    tabela["siafi"] = _na_se_vazio(tabela["codigo_siafi"].map(norm_siafi)).fillna("").astype(str)
    tabela.loc[tabela["ibge"].isin(["<NA>", "nan", "None"]), "ibge"] = ""
    tabela.loc[tabela["siafi"].isin(["<NA>", "nan", "None"]), "siafi"] = ""
    tabela["ufn"] = tabela["uf"].map(norm_uf)
    tabela["chave_nome"] = tabela["nome_municipio"].map(norm_nome_chave) + "|" + tabela["ufn"]
    ibge = {}
    siafi = {}
    for _, linha in tabela.dropna(subset=["ibge"]).iterrows():
        ibge[linha["ibge"]] = (linha["ibge"], linha["siafi"])
        if linha["siafi"]:
            siafi.setdefault(linha["siafi"], (linha["ibge"], linha["siafi"]))
    ibge6 = {}
    for codigo, par in ibge.items():
        if len(codigo) == 7:
            ibge6.setdefault(codigo[:6], par)
    # prefixo de 6 dígitos ambíguo não serve para casar
    contagem6: dict[str, int] = {}
    for codigo in ibge:
        if len(codigo) == 7:
            contagem6[codigo[:6]] = contagem6.get(codigo[:6], 0) + 1
    ibge6 = {k: v for k, v in ibge6.items() if contagem6.get(k) == 1}
    nome = {}
    for _, linha in tabela.iterrows():
        chave = linha["chave_nome"]
        if not chave or chave.startswith("|") or chave in nome:
            if chave in nome and nome[chave] != (linha["ibge"], linha["siafi"]):
                nome[chave] = None  # ambíguo
            continue
        nome[chave] = (linha["ibge"], linha["siafi"])
    nome = {k: v for k, v in nome.items() if v}
    alias = {
        (uf, norm_nome_chave(nome_tesouro)): ibge_alvo
        for uf, nome_tesouro, ibge_alvo in _ALIAS_TESOURO_IBGE
    }
    return {"ibge": ibge, "ibge6": ibge6, "siafi": siafi, "nome": nome, "alias": alias}


def atribuir_codigos_transferencias(df: pd.DataFrame, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Preenche codigo_ibge (7 dígitos) e codigo_siafi.

    O CSV mensal do Tesouro (Município, UF, decêndios, tipo) não traz código.
    Se aparecer uma coluna de código, ela é normalizada ('.0', zeros à esquerda,
    IBGE de 6 dígitos → 7). O que sobra casa nome+UF no crosswalk.
    """
    saida = df.copy()
    mapas = _mapas_crosswalk(crosswalk)
    colunas_entrada = [str(c) for c in df.columns]
    coluna = _coluna_codigo_municipio(saida)
    valores_codigo = saida[coluna].copy() if coluna else None
    saida["codigo_ibge"] = pd.NA
    saida["codigo_siafi"] = pd.NA
    nativos = 0
    if coluna and valores_codigo is not None:
        classes = valores_codigo.map(_classificar_codigo_bruto)
        tipos = classes.map(lambda par: par[0])
        contagem = tipos[tipos.ne("")].value_counts().to_dict()
        print(f"Transferências: coluna de código '{coluna}' ({contagem or 'sem valores numéricos'})")
        for indice, (tipo, chave) in classes.items():
            par = None
            if tipo == "ibge":
                par = mapas["ibge"].get(chave)
            elif tipo == "ibge6":
                par = mapas["ibge6"].get(chave)
            elif tipo == "siafi":
                par = mapas["siafi"].get(chave)
            if par and par[0]:
                saida.at[indice, "codigo_ibge"] = par[0]
                saida.at[indice, "codigo_siafi"] = par[1] or pd.NA
                nativos += 1
    else:
        print(
            "Transferências: sem coluna de código de município "
            f"(colunas: {', '.join(colunas_entrada)}). "
            "codigo_ibge/SIAFI saem do crosswalk pelo nome+UF."
        )
    nome_col = "nome_transf" if "nome_transf" in saida.columns else _coluna(saida, "municipio", "nome_municipio", "nome")
    uf_col = "uf_transf" if "uf_transf" in saida.columns else _coluna(saida, "uf")
    via_nome = 0
    if nome_col and uf_col:
        faltam = saida["codigo_ibge"].isna()
        chaves = saida.loc[faltam, nome_col].map(norm_nome_chave) + "|" + saida.loc[faltam, uf_col].map(norm_uf)
        for indice, chave in chaves.items():
            par = mapas["nome"].get(chave)
            if par is None and "|" in chave:
                uf = chave.rsplit("|", 1)[1]
                nome = chave.rsplit("|", 1)[0]
                ibge_alias = mapas["alias"].get((uf, nome))
                if ibge_alias:
                    par = mapas["ibge"].get(norm_ibge(ibge_alias))
            if par and par[0]:
                saida.at[indice, "codigo_ibge"] = par[0]
                saida.at[indice, "codigo_siafi"] = par[1] or pd.NA
                via_nome += 1
    if "codigo_ibge" in saida.columns:
        com = saida.dropna(subset=["codigo_ibge"])
        sem = saida[saida["codigo_ibge"].isna()]
        if not com.empty:
            agregacoes = {"transferencia_liquida": "sum"}
            if "codigo_siafi" in com.columns:
                agregacoes["codigo_siafi"] = "first"
            if nome_col and nome_col in com.columns:
                agregacoes[nome_col] = "first"
            if uf_col and uf_col in com.columns:
                agregacoes[uf_col] = "first"
            com = com.groupby("codigo_ibge", dropna=False, as_index=False).agg(agregacoes)
        saida = pd.concat([com, sem], ignore_index=True)
    print(
        f"Transferências: {nativos} municípios pelo código da própria base, "
        f"{via_nome} pelo crosswalk (nome+UF), "
        f"{int(saida['codigo_ibge'].isna().sum()) if 'codigo_ibge' in saida.columns else 0} sem código"
    )
    return saida


def carregar_transferencias(crosswalk: pd.DataFrame | None = None) -> pd.DataFrame:
    caminho = DATA_SOCIO / "transferencias_municipios.csv"
    if not caminho.exists():
        print(f"AVISO: {caminho} não encontrado")
        return pd.DataFrame()
    df = _ler_csv(caminho)
    pronta = _coluna(df, "transferencia_liquida", "transferencia liquida")
    decendios = [c for c in df.columns if "decendio" in _sem_acento(c)]
    col_valor = _coluna(df, "valor")
    if pronta:
        total = _numero(df[pronta])
    elif decendios:
        total = sum((_numero(df[c]).fillna(0) for c in decendios), start=0)
        print(f"Transferências: somando {len(decendios)} decêndios por município")
    elif col_valor and "decendio" not in _sem_acento(col_valor):
        total = _numero(df[col_valor])
    else:
        print(f"AVISO: {caminho} sem valor de transferência reconhecível")
        return pd.DataFrame()
    saida = pd.DataFrame({"transferencia_liquida": total})
    col_nome = _coluna(df, "municipio", "nome_municipio", "nome")
    col_uf = _coluna(df, "uf")
    col_codigo = _coluna_codigo_municipio(df)
    chaves = []
    if col_nome:
        saida["nome_transf"] = df[col_nome]
        chaves.append("nome_transf")
    if col_uf:
        saida["uf_transf"] = df[col_uf].map(norm_uf)
        chaves.append("uf_transf")
    if col_codigo and col_codigo not in (col_nome, col_uf):
        # Valor cru (pode vir como 1200401.0, 120040 ou 0139). A normalização é depois.
        saida["codigo_origem"] = df[col_codigo].astype(str)
        chaves.append("codigo_origem")
    if chaves:
        saida = saida.groupby(chaves, dropna=False, as_index=False)["transferencia_liquida"].sum()
    saida = atribuir_codigos_transferencias(saida, crosswalk if crosswalk is not None else pd.DataFrame())
    print(f"Transferências carregadas: {len(saida)} municípios")
    return saida


def excluir_exterior(municipios: pd.DataFrame) -> pd.DataFrame:
    """Tira do cruzamento a UF ZZ e códigos TSE sem IBGE fora do Brasil."""
    if municipios.empty:
        return municipios
    uf = municipios["sg_uf"].map(norm_uf)
    sem_ibge = municipios["codigo_ibge"].isna() if "codigo_ibge" in municipios.columns else pd.Series(True, index=municipios.index)
    exterior = uf.eq("ZZ") | (~uf.isin(UFS_BRASILEIRAS) & sem_ibge)
    if not exterior.any():
        print("Exterior: nenhum município ZZ excluído")
        return municipios
    fora = municipios.loc[exterior]
    secoes = int(pd.to_numeric(fora["n_secoes"], errors="coerce").fillna(0).sum()) if "n_secoes" in fora.columns else 0
    eleitores = int(pd.to_numeric(fora["n_eleitores"], errors="coerce").fillna(0).sum()) if "n_eleitores" in fora.columns else 0
    print(
        f"Exterior excluído: {len(fora)} municípios, {secoes} seções, {eleitores} eleitores "
        "(UF ZZ ou código TSE sem IBGE brasileiro)"
    )
    return municipios.loc[~exterior].reset_index(drop=True)


def _mes_ordinal(valor) -> float:
    texto = "" if valor is None else str(valor).strip()
    achado = re.search(r"(20\d{2})\D?(0[1-9]|1[0-2])", texto)
    if achado:
        return float(achado.group(1) + achado.group(2))
    invertido = re.search(r"(0[1-9]|1[0-2])\D?(20\d{2})", texto)
    if invertido:
        return float(invertido.group(2) + invertido.group(1))
    digitos = re.sub(r"\D", "", texto)
    if len(digitos) >= 6:
        return float(digitos[:6])
    return np.nan


def carregar_bolsa() -> pd.DataFrame:
    """Um registro por município: o mes_referencia mais recente."""
    caminho = DATA_SOCIO / "bolsa_familia_municipios.csv"
    if not caminho.exists():
        print(f"AVISO: {caminho} não encontrado")
        return pd.DataFrame()
    df = _ler_csv(caminho)
    col_codigo = _coluna(df, "codigo_municipio", "codigo_siafi", "codigo municipio siafi", "siafi")
    col_nome = _coluna(df, "nome_municipio", "municipio", "nome")
    col_uf = _coluna(df, "uf")
    col_mes = _coluna(df, "mes_referencia", "mes_competencia", "mes referencia")
    col_n = _coluna(df, "n_beneficiarios", "beneficiarios")
    col_valor = _coluna(df, "valor_parcela", "bolsa_familia_valor_ou_qtd", "valor")
    saida = pd.DataFrame(index=df.index)
    if col_codigo:
        saida["codigo_siafi"] = _na_se_vazio(df[col_codigo].map(norm_siafi))
    if col_nome:
        saida["nome_bolsa"] = df[col_nome]
    if col_uf:
        saida["uf_bolsa"] = df[col_uf].map(norm_uf)
    if col_mes:
        saida["bolsa_mes_referencia"] = df[col_mes].astype(str).str.strip()
        saida["mes_ord"] = df[col_mes].map(_mes_ordinal)
    else:
        saida["bolsa_mes_referencia"] = pd.NA
        saida["mes_ord"] = np.nan
        print("AVISO: Bolsa Família sem mes_referencia; agregando todas as linhas do município")
    if col_n:
        saida["bolsa_n_beneficiarios"] = _numero(df[col_n])
    if col_valor:
        saida["bolsa_valor_parcela"] = _numero(df[col_valor])
    agregacoes = {"bolsa_mes_referencia": "first"}
    if "bolsa_n_beneficiarios" in saida.columns:
        agregacoes["bolsa_n_beneficiarios"] = "sum"
    if "bolsa_valor_parcela" in saida.columns:
        agregacoes["bolsa_valor_parcela"] = "sum"
    partes = []
    if "codigo_siafi" in saida.columns and saida["codigo_siafi"].notna().any():
        com_codigo = saida[saida["codigo_siafi"].notna()].copy()
        if com_codigo["mes_ord"].notna().any():
            maximo = com_codigo.groupby("codigo_siafi", dropna=False)["mes_ord"].transform("max")
            com_codigo = com_codigo.loc[com_codigo["mes_ord"].eq(maximo)].copy()
        partes.append(com_codigo.groupby("codigo_siafi", dropna=False, as_index=False).agg(agregacoes))
    sem_codigo = saida[saida["codigo_siafi"].isna()].copy() if "codigo_siafi" in saida.columns else saida.copy()
    if "codigo_siafi" not in saida.columns:
        sem_codigo = saida.copy()
    if not sem_codigo.empty:
        chaves_nome = [c for c in ("uf_bolsa", "nome_bolsa") if c in sem_codigo.columns]
        if chaves_nome:
            if sem_codigo["mes_ord"].notna().any():
                maximo = sem_codigo.groupby(chaves_nome, dropna=False)["mes_ord"].transform("max")
                sem_codigo = sem_codigo.loc[sem_codigo["mes_ord"].eq(maximo) | sem_codigo["mes_ord"].isna()].copy()
            partes.append(sem_codigo.groupby(chaves_nome, dropna=False, as_index=False).agg(agregacoes))
    if not partes:
        print(f"Bolsa Família carregada: {len(saida)} linhas (sem chave municipal)")
        return saida
    if saida["mes_ord"].notna().any():
        print("Bolsa Família: mês de referência mais recente por município (não a média entre meses)")
    reduzida = pd.concat(partes, ignore_index=True)
    print(f"Bolsa Família carregada: {len(reduzida)} municípios (mês mais recente)")
    return reduzida


def _juntar_por_codigo_e_nome(
    base: pd.DataFrame,
    extra: pd.DataFrame,
    chave_codigo: str,
    nome_extra: str,
    uf_extra: str,
    colunas_valor: list[str],
    nome_match: str,
) -> pd.DataFrame:
    """Junta por código e, no que sobrar, por nome+UF normalizado."""
    saida = base.copy()
    saida[nome_match] = "sem_match"
    colunas_valor = [c for c in colunas_valor if c in extra.columns]
    if extra.empty or not colunas_valor:
        for coluna in colunas_valor:
            saida[coluna] = np.nan
        return saida
    if chave_codigo and chave_codigo in extra.columns and chave_codigo in saida.columns:
        pedaco = extra.dropna(subset=[chave_codigo]).drop_duplicates(chave_codigo)
        saida = saida.merge(pedaco[[chave_codigo, *colunas_valor]], on=chave_codigo, how="left")
        saida.loc[saida[colunas_valor[0]].notna(), nome_match] = "codigo"
    else:
        for coluna in colunas_valor:
            if coluna not in saida.columns:
                saida[coluna] = np.nan

    faltam = saida[nome_match].eq("sem_match")
    if not faltam.any() or not nome_extra or nome_extra not in extra.columns:
        return saida
    tabela = extra.copy()
    tem_uf = bool(uf_extra) and uf_extra in tabela.columns and tabela[uf_extra].map(norm_uf).ne("").any()
    if tem_uf:
        tabela["chave"] = tabela[nome_extra].map(norm_nome) + "|" + tabela[uf_extra].map(norm_uf)
        chaves_base = saida.loc[faltam, "nome_municipio"].map(norm_nome) + "|" + saida.loc[faltam, "sg_uf"].map(norm_uf)
    else:
        print(f"AVISO: {nome_match} sem UF; nome sozinho só quando é único no país")
        tabela["chave"] = tabela[nome_extra].map(norm_nome)
        chaves_base = saida.loc[faltam, "nome_municipio"].map(norm_nome)
        repetidos = chaves_base.value_counts()
        repetidos = set(repetidos[repetidos > 1].index)
        chaves_base = chaves_base.mask(chaves_base.isin(repetidos), "")
    tabela = tabela[tabela["chave"].str.len().gt(0) & ~tabela["chave"].str.startswith("|")]
    tabela = tabela.drop_duplicates("chave", keep=False)
    if tabela.empty:
        return saida
    mapa = tabela.set_index("chave")
    uteis = chaves_base[chaves_base.isin(mapa.index) & chaves_base.ne("") & ~chaves_base.str.startswith("|")]
    if uteis.empty:
        return saida
    escolhidos = mapa.loc[uteis.to_numpy()]
    for coluna in colunas_valor:
        if coluna in escolhidos.columns:
            saida.loc[uteis.index, coluna] = escolhidos[coluna].to_numpy()
    saida.loc[uteis.index, nome_match] = "nome"
    return saida


def _taxa(numerador: pd.Series, denominador: pd.Series) -> pd.Series:
    den = _numero(denominador)
    num = _numero(numerador)
    return num / den.where(den > 0)


def _log_match(rotulo: str, serie: pd.Series) -> None:
    n = len(serie)
    if n == 0:
        print(f"{rotulo}: sem municípios")
        return
    por_codigo = int((serie == "codigo").sum())
    por_nome = int((serie == "nome").sum())
    sem = n - por_codigo - por_nome
    print(
        f"{rotulo}: {por_codigo}/{n} por código ({por_codigo / n:.1%}), "
        f"{por_nome}/{n} por nome+UF ({por_nome / n:.1%}), {sem}/{n} sem match"
    )


COLUNAS_GASTO_PC = (
    "siconfi_despesa_total",
    "siconfi_despesa_saude",
    "siconfi_despesa_educacao",
    "siconfi_despesa_assistencia",
    "siconfi_despesa_urbanismo",
    "siconfi_receita_propria",
    "siconfi_receita_transferida",
    "siconfi_receita_fundeb",
    "emendas_valor_pago_eleicao",
    "emendas_valor_pago_ano_anterior",
    "emendas_valor_pago_bienio",
    "siops_despesa",
    "siope_despesa",
    "fundeb_repasse",
    "convenio_valor",
    "despesa_federal_valor",
)

LEGENDA_GASTO = (
    "siconfi_despesa_total: despesas liquidadas exceto intraorçamentárias (Siconfi DCA Anexo I-E).",
    "siconfi_despesa_saude: função 10 Saúde, despesas liquidadas (DCA I-E).",
    "siconfi_despesa_educacao: função 12 Educação, despesas liquidadas (DCA I-E).",
    "siconfi_despesa_assistencia: função 08 Assistência Social, despesas liquidadas (DCA I-E).",
    "siconfi_despesa_urbanismo: função 15 Urbanismo, despesas liquidadas (DCA I-E).",
    "siconfi_receita_propria: impostos, taxas e contribuições de melhoria, receitas brutas (DCA I-C, 1.1.0.0.00.0.0).",
    "siconfi_receita_transferida: transferências correntes, receitas brutas (DCA I-C, 1.7.0.0.00.0.0).",
    "siconfi_receita_fundeb: FUNDEB recebido pelo município no DCA (1.7.5.8.01.0.0), não o arquivo de repasse do FNDE.",
    "siconfi_populacao: população informada pelo Siconfi no exercício do DCA.",
    "emendas_valor_pago_eleicao: soma do valor pago das emendas no ano da eleição (Portal da Transparência, EmendasParlamentares.csv).",
    "emendas_valor_pago_ano_anterior: idem, ano anterior à eleição.",
    "emendas_valor_pago_bienio: soma dos dois anos.",
    "siops_despesa: despesa SIOPS lida de arquivo local, se houver.",
    "siope_despesa: despesa SIOPE lida de arquivo local, se houver.",
    "fundeb_repasse: repasse FNDE/FUNDEB lido de arquivo local, se houver.",
    "convenio_valor: transferências voluntárias/convênios lidas de arquivo local, se houver.",
    "despesa_federal_valor: despesa federal por localidade lida de arquivo local agregado, se houver.",
    "Colunas *_pc: valor dividido pela população do Siconfi no ano do DCA, ou pela pop do IDHM 2010 se a do Siconfi faltar.",
)


def _populacao_para_gasto(tabela: pd.DataFrame) -> pd.Series:
    pop = pd.Series(np.nan, index=tabela.index, dtype="float64")
    if "siconfi_populacao" in tabela.columns:
        pop = _numero(tabela["siconfi_populacao"])
    if "pop" in tabela.columns:
        pop = pop.where(pop > 0, _numero(tabela["pop"]))
    return pop


def _per_capita_gastos(tabela: pd.DataFrame) -> pd.DataFrame:
    pop = _populacao_para_gasto(tabela)
    for coluna in COLUNAS_GASTO_PC:
        if coluna in tabela.columns:
            tabela[coluna + "_pc"] = _taxa(tabela[coluna], pop)
    return tabela


def anexar_gastos_publicos(municipios: pd.DataFrame, ano: str, crosswalk: pd.DataFrame) -> pd.DataFrame:
    """Siconfi, emendas e bases locais. Falha de uma fonte não interrompe as outras."""
    codigos = []
    if "codigo_ibge" in municipios.columns:
        codigos = [c for c in municipios["codigo_ibge"].map(norm_ibge) if c and len(c) == 7]
    piloto = "1200401" if "1200401" in codigos else (codigos[0] if codigos else "1200401")
    try:
        exercicio = gastos.escolher_exercicio_siconfi(int(ano), piloto)
        siconfi = gastos.garantir_siconfi(exercicio, codigos, pasta=DATA_SOCIO) if exercicio else pd.DataFrame()
    except Exception as exc:
        print(f"Siconfi: falhou ({exc}). Seguindo sem DCA.")
        siconfi = pd.DataFrame()
    if siconfi is not None and not siconfi.empty:
        ordem = [
            "siconfi_despesa_total", "siconfi_despesa_saude", "siconfi_despesa_educacao",
            "siconfi_despesa_assistencia", "siconfi_despesa_urbanismo",
            "siconfi_receita_propria", "siconfi_receita_transferida", "siconfi_receita_fundeb",
            "siconfi_populacao", "siconfi_exercicio",
        ]
        colunas = [c for c in ordem if c in siconfi.columns]
        municipios = _juntar_por_codigo_e_nome(municipios, siconfi, "codigo_ibge", "", "", colunas, "match_siconfi")
        ano_dca = siconfi["siconfi_exercicio"].dropna().astype(str).iloc[0] if "siconfi_exercicio" in siconfi.columns and siconfi["siconfi_exercicio"].notna().any() else "?"
        _log_match(f"Siconfi DCA {ano_dca}", municipios["match_siconfi"])
    else:
        print("Siconfi: sem dados; despesa por função fica de fora deste ano")

    try:
        bruto = gastos.garantir_emendas(pasta=DATA_SOCIO)
        emendas = gastos.emendas_para_eleicao(bruto, int(ano))
    except Exception as exc:
        print(f"Emendas: falhou ({exc}). Seguindo sem emendas.")
        emendas = pd.DataFrame()
    if emendas is not None and not emendas.empty:
        colunas = [c for c in emendas.columns if c != "codigo_ibge"]
        municipios = _juntar_por_codigo_e_nome(municipios, emendas, "codigo_ibge", "", "", colunas, "match_emendas")
        _log_match(f"Emendas {ano} e {int(ano) - 1}", municipios["match_emendas"])
    else:
        print("Emendas: sem dados para este cruzamento")

    for fonte in gastos.FONTES_MANUAIS:
        try:
            extra = gastos.carregar_fonte_manual(fonte)
        except Exception as exc:
            print(f"{fonte['rotulo']}: falhou ({exc}). Pulando.")
            continue
        if extra is None or extra.empty or fonte["coluna"] not in extra.columns:
            continue
        nome_match = "match_" + fonte["coluna"]
        municipios = _juntar_por_codigo_e_nome(
            municipios, extra, "codigo_ibge", "", "", [fonte["coluna"]], nome_match,
        )
        _log_match(fonte["rotulo"], municipios[nome_match])
    return municipios


def montar_cruzamento(ano: str) -> pd.DataFrame:
    secoes = calcular_residuais_secao(ano)
    if secoes.empty:
        return pd.DataFrame()
    municipios = agregar_residual_municipio(secoes)
    if municipios.empty:
        print("Não foi possível agregar residual por município. Verifique colunas nos joins.")
        return pd.DataFrame()
    print(
        f"Residual agregado: {len(municipios)} municípios, {int(municipios['n_secoes'].sum())} seções "
        "(média ponderada por comparecimento)"
    )
    crosswalk = carregar_crosswalk()
    municipios = anexar_crosswalk(municipios, crosswalk)
    municipios = excluir_exterior(municipios)
    if municipios.empty:
        print("Não foi possível agregar residual por município. Verifique colunas nos joins.")
        return pd.DataFrame()
    municipios["ano"] = str(ano)

    idh = carregar_idh()
    municipios = _juntar_por_codigo_e_nome(
        municipios,
        idh,
        "codigo_ibge",
        "nome_idh",
        "uf_idh",
        [c for c in ("idhm", "idhm_e", "idhm_l", "idhm_r", "pop") if c in idh.columns] or ["idhm"],
        "match_idh",
    )
    _log_match("IDHM", municipios["match_idh"])

    transf = carregar_transferencias(crosswalk)
    colunas_transf = ["transferencia_liquida"]
    if "codigo_ibge" in transf.columns and transf["codigo_ibge"].notna().any():
        chave_transf = "codigo_ibge"
    elif "codigo_siafi" in transf.columns and transf["codigo_siafi"].notna().any():
        chave_transf = "codigo_siafi"
    else:
        chave_transf = ""
    municipios = _juntar_por_codigo_e_nome(
        municipios,
        transf,
        chave_transf,
        "nome_transf" if "nome_transf" in transf.columns else "",
        "uf_transf" if "uf_transf" in transf.columns else "",
        colunas_transf,
        "match_transferencias",
    )
    _log_match("Transferências", municipios["match_transferencias"])

    bolsa = carregar_bolsa()
    colunas_bolsa = [c for c in ("bolsa_mes_referencia", "bolsa_n_beneficiarios", "bolsa_valor_parcela") if c in bolsa.columns]
    municipios = _juntar_por_codigo_e_nome(
        municipios,
        bolsa,
        "codigo_siafi" if "codigo_siafi" in bolsa.columns else "",
        "nome_bolsa" if "nome_bolsa" in bolsa.columns else "",
        "uf_bolsa" if "uf_bolsa" in bolsa.columns else "",
        colunas_bolsa or ["bolsa_valor_parcela"],
        "match_bolsa",
    )
    _log_match("Bolsa Família", municipios["match_bolsa"])

    municipios = anexar_gastos_publicos(municipios, str(ano), crosswalk)

    if "pop" in municipios.columns and "transferencia_liquida" in municipios.columns:
        municipios["transferencia_per_capita"] = _taxa(municipios["transferencia_liquida"], municipios["pop"])
    if "pop" in municipios.columns and "bolsa_valor_parcela" in municipios.columns:
        municipios["bolsa_valor_per_capita"] = _taxa(municipios["bolsa_valor_parcela"], municipios["pop"])
    if "pop" in municipios.columns and "bolsa_n_beneficiarios" in municipios.columns:
        municipios["bolsa_beneficiarios_por_hab"] = _taxa(municipios["bolsa_n_beneficiarios"], municipios["pop"])
    if "bolsa_n_beneficiarios" in municipios.columns:
        municipios["bolsa_beneficiarios_por_eleitor"] = _taxa(municipios["bolsa_n_beneficiarios"], municipios["n_eleitores"])
    municipios = _per_capita_gastos(municipios)
    if "pop" not in municipios.columns:
        print("Sem população do IDHM: medidas per capita de transferência e Bolsa ficam vazias; a razão por eleitor usa n_eleitores")
    return municipios


def _escrever_correlacao(ano: str, tabela: pd.DataFrame, destino: Path) -> None:
    foco = [
        "idhm", "idhm_e", "idhm_l", "idhm_r", "pop",
        "transferencia_liquida", "transferencia_per_capita",
        "bolsa_valor_parcela", "bolsa_valor_per_capita",
        "bolsa_n_beneficiarios", "bolsa_beneficiarios_por_hab", "bolsa_beneficiarios_por_eleitor",
        "siconfi_despesa_total", "siconfi_despesa_total_pc",
        "siconfi_despesa_saude", "siconfi_despesa_saude_pc",
        "siconfi_despesa_educacao", "siconfi_despesa_educacao_pc",
        "siconfi_despesa_assistencia", "siconfi_despesa_assistencia_pc",
        "siconfi_despesa_urbanismo", "siconfi_despesa_urbanismo_pc",
        "siconfi_receita_propria", "siconfi_receita_propria_pc",
        "siconfi_receita_transferida", "siconfi_receita_transferida_pc",
        "siconfi_receita_fundeb", "siconfi_receita_fundeb_pc",
        "emendas_valor_pago_eleicao", "emendas_valor_pago_eleicao_pc",
        "emendas_valor_pago_ano_anterior", "emendas_valor_pago_ano_anterior_pc",
        "emendas_valor_pago_bienio", "emendas_valor_pago_bienio_pc",
        "siops_despesa", "siops_despesa_pc",
        "siope_despesa", "siope_despesa_pc",
        "fundeb_repasse", "fundeb_repasse_pc",
        "convenio_valor", "convenio_valor_pc",
        "despesa_federal_valor", "despesa_federal_valor_pc",
        "n_eleitores", "n_comparecimento", "n_secoes", "residual_mean_unweighted",
    ]
    presentes = []
    for coluna in foco:
        if coluna not in tabela.columns:
            continue
        if _numero(tabela[coluna]).notna().sum() >= 3:
            presentes.append(coluna)
    linhas = [
        f"Correlações com residual_mean ({ano})",
        "=" * 50,
        "residual_mean é a média municipal do residual de seção (observado − previsto, p.p.),",
        "ponderada por QT_COMPARECIMENTO_t2. residual_mean_unweighted é a média simples.",
        "O residual é o de process_and_summarize.residual_distribuicao (regressão linear T1→T2,",
        "seções de teste com NR_SECAO não múltiplo de 3).",
        "Bolsa Família: mes_referencia mais recente de cada município, não a média dos meses.",
        "Crosswalk: betafcc/Municipios-Brasileiros-TSE + kelvins/municipios-brasileiros (SIAFI),",
        "arquivo data/referencia/municipios_tse_ibge_siafi.csv.",
        "Transferências: o CSV mensal do Tesouro não tem coluna de código (Município, UF, decêndios, tipo).",
        "O codigo_ibge usado no cruzamento é o do crosswalk (nome+UF, com DOeste/D'Oeste unificados).",
        "Se o arquivo trouxer código, ele é normalizado (sem '.0', SIAFI com zeros, IBGE de 6 dígitos casa nos 7).",
        "Municípios da UF ZZ (exterior) ficam fora do cruzamento e das correlações.",
        "Per capita de IDH/Bolsa/transferência constitucional usa a pop do IDHM 2010.",
        "Per capita de gasto (Siconfi, emendas e bases locais) usa a população do Siconfi no ano do DCA e, se faltar, a pop do IDHM 2010.",
        "Legenda das variáveis de gasto:",
        *LEGENDA_GASTO,
        "Fontes não baixadas automaticamente: SIOPS, SIOPE, repasse FNDE/FUNDEB, Transferegov/convênios e despesa federal por localidade.",
        "Motivo: não há API em massa sem autenticação (ou o arquivo do portal é HTML/403 ou multi-GB). Se o CSV estiver em SOCIO_LOCAL_DIR, fora de raw/, ele entra.",
        "",
    ]
    if not presentes or "residual_mean" not in tabela.columns:
        linhas.append("Sem colunas numéricas suficientes para correlação.")
    else:
        sub = tabela[["residual_mean", *presentes]].apply(_numero)
        correlacoes = sub.corr(numeric_only=True)["residual_mean"].drop(labels=["residual_mean"]).sort_values(ascending=False)
        linhas.append(correlacoes.to_string())
        print(correlacoes.head(10).to_string())
    linhas.append("")
    linhas.append(f"N municípios: {len(tabela)}")
    for coluna, rotulo in (
        ("match_crosswalk", "crosswalk"),
        ("match_idh", "IDHM"),
        ("match_transferencias", "transferências"),
        ("match_bolsa", "Bolsa Família"),
        ("match_siconfi", "Siconfi"),
        ("match_emendas", "emendas"),
        ("match_siops_despesa", "SIOPS"),
        ("match_siope_despesa", "SIOPE"),
        ("match_fundeb_repasse", "FUNDEB/FNDE"),
        ("match_convenio_valor", "convênios"),
        ("match_despesa_federal_valor", "despesa federal"),
    ):
        if coluna in tabela.columns:
            contagem = tabela[coluna].value_counts(dropna=False).to_dict()
            linhas.append(f"Match {rotulo}: {contagem}")
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"Resumo salvo em {destino}")


def _ordenar_colunas(tabela: pd.DataFrame) -> pd.DataFrame:
    ordem = [
        "ano", "sg_uf", "cd_municipio_tse", "codigo_ibge", "codigo_siafi", "nome_municipio",
        "n_secoes", "n_eleitores", "n_comparecimento",
        "residual_mean", "residual_mean_unweighted", "residual_std",
        "idhm", "idhm_e", "idhm_l", "idhm_r", "pop",
        "transferencia_liquida", "transferencia_per_capita",
        "bolsa_mes_referencia", "bolsa_n_beneficiarios", "bolsa_valor_parcela",
        "bolsa_valor_per_capita", "bolsa_beneficiarios_por_hab", "bolsa_beneficiarios_por_eleitor",
        "siconfi_exercicio", "siconfi_populacao",
        "siconfi_despesa_total", "siconfi_despesa_saude", "siconfi_despesa_educacao",
        "siconfi_despesa_assistencia", "siconfi_despesa_urbanismo",
        "siconfi_receita_propria", "siconfi_receita_transferida", "siconfi_receita_fundeb",
        "siconfi_despesa_total_pc", "siconfi_despesa_saude_pc", "siconfi_despesa_educacao_pc",
        "siconfi_despesa_assistencia_pc", "siconfi_despesa_urbanismo_pc",
        "siconfi_receita_propria_pc", "siconfi_receita_transferida_pc", "siconfi_receita_fundeb_pc",
        "emendas_valor_pago_eleicao", "emendas_valor_pago_ano_anterior", "emendas_valor_pago_bienio",
        "emendas_valor_pago_eleicao_pc", "emendas_valor_pago_ano_anterior_pc", "emendas_valor_pago_bienio_pc",
        "siops_despesa", "siops_despesa_pc", "siope_despesa", "siope_despesa_pc",
        "fundeb_repasse", "fundeb_repasse_pc", "convenio_valor", "convenio_valor_pc",
        "despesa_federal_valor", "despesa_federal_valor_pc",
        "match_crosswalk", "match_idh", "match_transferencias", "match_bolsa",
        "match_siconfi", "match_emendas", "match_siops_despesa", "match_siope_despesa",
        "match_fundeb_repasse", "match_convenio_valor", "match_despesa_federal_valor",
    ]
    existentes = [c for c in ordem if c in tabela.columns]
    resto = [c for c in tabela.columns if c not in existentes]
    saida = tabela[existentes + resto].copy()
    if "codigo_siafi" in saida.columns:
        saida["codigo_siafi"] = saida["codigo_siafi"].map(lambda v: norm_siafi(v).zfill(4) if norm_siafi(v) else pd.NA)
    if "codigo_ibge" in saida.columns:
        saida["codigo_ibge"] = saida["codigo_ibge"].map(lambda v: norm_ibge(v) if norm_ibge(v) else pd.NA)
    for coluna in ("residual_mean", "residual_mean_unweighted", "residual_std"):
        if coluna in saida.columns:
            saida[coluna] = _numero(saida[coluna]).round(6)
    if "bolsa_n_beneficiarios" in saida.columns:
        saida["bolsa_n_beneficiarios"] = _numero(saida["bolsa_n_beneficiarios"]).round().astype("Int64")
    for coluna in saida.columns:
        if coluna.endswith("_pc") or coluna in ("transferencia_per_capita", "bolsa_valor_per_capita", "bolsa_beneficiarios_por_hab", "bolsa_beneficiarios_por_eleitor"):
            saida[coluna] = _numero(saida[coluna]).round(6)
        elif coluna in ("transferencia_liquida", "bolsa_valor_parcela") or coluna.startswith((
            "siconfi_despesa", "siconfi_receita", "emendas_valor", "siops_", "siope_", "fundeb_", "convenio_", "despesa_federal",
        )):
            saida[coluna] = _numero(saida[coluna]).round(2)
    if "nome_municipio" in saida.columns:
        saida = saida.sort_values(["sg_uf", "nome_municipio"], kind="mergesort")
    return saida.reset_index(drop=True)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ano", default="2026")
    args = parser.parse_args()
    print(f"Cruzamento socioeconômico para {args.ano}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    tabela = montar_cruzamento(str(args.ano))
    if tabela.empty:
        print("Sem dados de residual agregado.")
        return
    tabela = _ordenar_colunas(tabela)
    destino = RESULTS / f"residual_socio_{args.ano}.csv"
    tabela.to_csv(destino, index=False, encoding="utf-8")
    print(f"Resultado salvo em {destino} ({len(tabela)} municípios)")
    _escrever_correlacao(str(args.ano), tabela, RESULTS / f"correlacao_socio_{args.ano}.txt")


if __name__ == "__main__":
    main()
