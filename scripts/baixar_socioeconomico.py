#!/usr/bin/env python3
"""
Baixa e prepara bases oficiais para cruzamento socioeconômico:
- IDHM municipal (base Atlas Brasil — último censo completo 2010)
- Transferências constitucionais mensais (Tesouro Transparente)
- Bolsa Família / Novo Bolsa Família — arquivo local (SOCIO_LOCAL_DIR) ou Portal da Transparência
- Nota sobre IVS (Ipea)

Nota sobre IDHM:
  O último IDHM municipal completo e público em CSV é o de 2010 (Censo).
  O Radar IDHM 2024 (2012-2024, PNAD Contínua) cobre Brasil, UFs e Regiões
  Metropolitanas, mas NÃO o nível municipal individual.

Salva em data/socioeconomico/
"""

from __future__ import annotations

from pathlib import Path
import csv
import functools
import io
import os
import re
import tempfile
import unicodedata
import zipfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests

# O runner bufferiza stdout; cada print sai na hora para o log do workflow.
print = functools.partial(print, flush=True)

DATA_SOCIO = Path("data/socioeconomico")
DATA_SOCIO.mkdir(parents=True, exist_ok=True)

_UF_POR_CODIGO_IBGE = {
    11: "RO", 12: "AC", 13: "AM", 14: "RR", 15: "PA", 16: "AP", 17: "TO",
    21: "MA", 22: "PI", 23: "CE", 24: "RN", 25: "PB", 26: "PE", 27: "AL", 28: "SE", 29: "BA",
    31: "MG", 32: "ES", 33: "RJ", 35: "SP",
    41: "PR", 42: "SC", 43: "RS",
    50: "MS", 51: "MT", 52: "GO", 53: "DF",
}


def _coluna_por_nome(df: pd.DataFrame, *opcoes: str) -> str | None:
    indice = {}
    for coluna in df.columns:
        chave = _sem_acento(str(coluna))
        indice.setdefault(chave, coluna)
        indice.setdefault(chave.replace(" ", ""), coluna)
    for opcao in opcoes:
        chave = _sem_acento(opcao)
        if chave in indice:
            return indice[chave]
        if chave.replace(" ", "") in indice:
            return indice[chave.replace(" ", "")]
    return None


def _numero_um(valor):
    if valor is None or (isinstance(valor, float) and pd.isna(valor)):
        return np.nan
    if isinstance(valor, (int, float)) and not isinstance(valor, bool):
        return float(valor)
    texto = str(valor).strip()
    if texto.lower() in {"", "nan", "none", "<na>"}:
        return np.nan
    if "," in texto:
        texto = texto.replace(".", "").replace(",", ".")
    elif re.fullmatch(r"\d{1,3}(\.\d{3})+", texto) and (texto.count(".") > 1 or texto.endswith(".000")):
        texto = texto.replace(".", "")
    numero = pd.to_numeric(texto, errors="coerce")
    return float(numero) if pd.notna(numero) else np.nan


def _numero_flex(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")
    return serie.map(_numero_um)


def _uf_sigla(valor) -> str:
    texto = str(valor).strip()
    if re.fullmatch(r"[A-Za-z]{2}", texto):
        return texto.upper()
    try:
        codigo = int(float(texto))
    except (TypeError, ValueError):
        return texto.upper()
    return _UF_POR_CODIGO_IBGE.get(codigo, texto.upper())


def preparar_idh(df: pd.DataFrame) -> pd.DataFrame:
    """Fica com o IDHM municipal de 2010 e o código IBGE de 7 dígitos (Codmun7)."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_ano = _coluna_por_nome(df, "ano")
    col_ibge = _coluna_por_nome(df, "codmun7", "codigo_ibge", "cd_mun", "cod mun", "codmun6")
    col_nome = _coluna_por_nome(df, "municipio", "nome municipio", "nome")
    col_uf = _coluna_por_nome(df, "uf")
    col_idhm = _coluna_por_nome(df, "idhm")
    col_e = _coluna_por_nome(df, "idhm_e", "idhm e")
    col_l = _coluna_por_nome(df, "idhm_l", "idhm l")
    col_r = _coluna_por_nome(df, "idhm_r", "idhm r")
    col_pop = _coluna_por_nome(df, "pop", "populacao")

    if col_ano:
        ano = pd.to_numeric(df[col_ano], errors="coerce")
        if (ano == 2010).any():
            df = df.loc[ano == 2010].copy()
            ano_uso = 2010
        else:
            ano_uso = int(ano.max()) if ano.notna().any() else None
            if ano_uso is not None:
                df = df.loc[ano == ano_uso].copy()
                print(f"AVISO: IDHM sem ano 2010; usando {ano_uso}")
        if ano_uso is not None:
            print(f"IDHM ano {ano_uso}: {len(df)} linhas")

    saida = pd.DataFrame(index=df.index)
    if col_ibge:
        codigo = df[col_ibge].astype(str).str.replace(r"\.0$", "", regex=True).str.replace(r"\D", "", regex=True)
        saida["codigo_ibge"] = codigo.map(lambda d: d.zfill(7) if d and len(d) != 6 else d)
    if col_nome:
        saida["municipio"] = df[col_nome].astype(str).str.strip()
    if col_uf:
        saida["uf"] = df[col_uf].map(_uf_sigla)
    for destino, origem in (("idhm", col_idhm), ("idhm_e", col_e), ("idhm_l", col_l), ("idhm_r", col_r), ("pop", col_pop)):
        if origem:
            saida[destino] = _numero_flex(df[origem])
    if col_ano and "ano" not in saida.columns:
        saida["ano"] = pd.to_numeric(df[col_ano], errors="coerce")
    saida = saida.dropna(how="all")
    if "codigo_ibge" in saida.columns:
        saida["codigo_ibge"] = saida["codigo_ibge"].replace("", pd.NA)
        saida = saida.dropna(subset=["codigo_ibge"]).drop_duplicates("codigo_ibge")
    return saida.reset_index(drop=True)


def preparar_transferencias(df: pd.DataFrame) -> pd.DataFrame:
    """Soma os decêndios (ou a coluna de valor) por município e UF.

    No CSV do Tesouro Transparente a coluna 'Transferência' é o tipo (FPM, FUNDEB),
    não o valor. O valor está em 1º/2º/3º Decêndio.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_pronta = _coluna_por_nome(df, "transferencia_liquida", "transferencia liquida")
    col_muni = _coluna_por_nome(df, "municipio", "nome municipio", "nome")
    col_uf = _coluna_por_nome(df, "uf")
    col_ibge = _coluna_por_nome(df, "codigo_ibge", "codmun7", "cd_mun")
    col_siafi = _coluna_por_nome(df, "codigo_siafi", "siafi", "codigo municipio siafi")
    if col_pronta:
        saida = pd.DataFrame()
        if col_muni:
            saida["municipio"] = df[col_muni]
        if col_uf:
            saida["uf"] = df[col_uf].map(_uf_sigla)
        saida["transferencia_liquida"] = _numero_flex(df[col_pronta])
        if col_ibge:
            saida["codigo_ibge"] = df[col_ibge]
        if col_siafi:
            saida["codigo_siafi"] = df[col_siafi]
        chaves = [c for c in ("municipio", "uf", "codigo_ibge", "codigo_siafi") if c in saida.columns]
        if not chaves:
            return saida
        return saida.groupby(chaves, dropna=False, as_index=False)["transferencia_liquida"].sum()

    decendios = [c for c in df.columns if "decendio" in _sem_acento(c)]
    col_valor = _coluna_por_nome(df, "valor")
    if decendios:
        total = sum((_numero_flex(df[c]).fillna(0) for c in decendios), start=0)
        print(f"Transferências: soma de {len(decendios)} colunas de decêndio")
    elif col_valor:
        total = _numero_flex(df[col_valor]).fillna(0)
    else:
        print("AVISO: transferências sem coluna de valor nem decêndio; arquivo mantido como veio")
        return df

    saida = pd.DataFrame({"transferencia_liquida": total})
    chaves = []
    if col_muni:
        saida["municipio"] = df[col_muni].astype(str).str.strip()
        chaves.append("municipio")
    if col_uf:
        saida["uf"] = df[col_uf].map(_uf_sigla)
        chaves.append("uf")
    if col_ibge:
        saida["codigo_ibge"] = df[col_ibge]
        chaves.append("codigo_ibge")
    if col_siafi:
        saida["codigo_siafi"] = df[col_siafi]
        chaves.append("codigo_siafi")
    if not chaves:
        return saida
    return saida.groupby(chaves, dropna=False, as_index=False)["transferencia_liquida"].sum()


def baixar_idh():
    url = "https://raw.githubusercontent.com/mauriciocramos/IDHM/master/municipal.csv"
    print(f"Baixando IDHM municipal (base 2010) de {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=";", encoding="utf-8", decimal=",", thousands=".", low_memory=False)
    df_out = preparar_idh(df)
    out_path = DATA_SOCIO / "idh_municipios.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"IDHM municipal (2010) salvo em {out_path} ({len(df_out)} municípios)")
    print("NOTA: Radar IDHM 2024 existe (PNAD 2012-2024) mas apenas para UF e RMs, não municipal.")
    return out_path

def baixar_transferencias():
    url = "https://www.tesourotransparente.gov.br/ckan/dataset/af4e7c47-2132-4d9a-bd7c-34e28a210b03/resource/315e91df-109d-41c2-b8c6-820eb3ffe0cd/download/transferenciamensalmunicipios202606.csv.csv"
    print(f"Baixando transferências de {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=";", encoding="latin1", low_memory=False)
    agg = preparar_transferencias(df)
    out_path = DATA_SOCIO / "transferencias_municipios.csv"
    agg.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Transferências salvas em {out_path} ({len(agg)} registros)")
    return out_path

def gerar_meses_recentes(n_meses: int = 12) -> list[str]:
    """Gera lista de YYYYMM dos últimos n meses (do mais recente para o mais antigo)."""
    hoje = datetime.now()
    meses = []
    for i in range(n_meses):
        data = hoje - timedelta(days=30 * i)
        meses.append(data.strftime("%Y%m"))
    return meses

# Arquivo oficial é um ZIP com um CSV por beneficiário (pode passar de vários GB).
BOLSA_CHUNK = 200_000
BOLSA_AMOSTRA = 256 * 1024
_YYYYMM = re.compile(r"(20\d{2})(0[1-9]|1[0-2])")
_SAIDA_BOLSA = "bolsa_familia_municipios.csv"


def _fmt_tamanho(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n} bytes ({n / (1024 * 1024):.1f} MiB)"
    return f"{n} bytes"


def _sem_acento(texto: str) -> str:
    base = unicodedata.normalize("NFKD", str(texto))
    base = "".join(ch for ch in base if not unicodedata.combining(ch))
    base = base.lower().replace("_", " ").replace("-", " ")
    return " ".join(base.split())


def _parece_texto_csv(cabeca: bytes) -> bool:
    if not cabeca:
        return False
    linha = cabeca.splitlines()[0]
    if b"<" in linha:
        return False
    return b";" in linha or linha.count(b",") >= 2


def _parece_html(conteudo: bytes, content_type: str = "") -> bool:
    """Página de erro do portal (HTTP 200 com HTML) não é um CSV."""
    dados = conteudo
    if dados.startswith(b"\xef\xbb\xbf"):
        dados = dados[3:]
    if dados.startswith(b"PK"):
        return False
    cabeca = dados[:1500].lstrip().lower()
    if cabeca.startswith((b"<!doctype", b"<html", b"<head", b"<body", b"<?xml")):
        return True
    if b"<html" in cabeca or b"<!doctype" in cabeca:
        return True
    tipo = (content_type or "").lower()
    if ("text/html" in tipo or "application/xhtml" in tipo) and not _parece_texto_csv(cabeca):
        return True
    return False


def _detectar_encoding(amostra: bytes) -> str:
    """latin-1 é o padrão do portal; utf-8 só quando a amostra é UTF-8 com acento."""
    if amostra.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig"
    try:
        texto = amostra.decode("utf-8")
    except UnicodeDecodeError:
        return "latin-1"
    if any(ord(ch) > 127 for ch in texto):
        return "utf-8"
    return "latin-1"


def _ordem_encodings(amostra: bytes) -> list[str]:
    preferido = _detectar_encoding(amostra)
    ordem = [preferido]
    for enc in ("latin-1", "utf-8", "utf-8-sig"):
        if enc not in ordem:
            ordem.append(enc)
    return ordem


def _detectar_sep(cabecalho: str) -> str:
    if cabecalho.count(";") >= 1:
        return ";"
    if cabecalho.count(",") >= 1:
        return ","
    if "\t" in cabecalho:
        return "\t"
    try:
        return csv.Sniffer().sniff(cabecalho, delimiters=";,\t|").delimiter
    except Exception:
        return ";"


def _detectar_decimal(texto: str, sep: str) -> str:
    if sep != ";":
        return "."
    linhas = texto.splitlines()[1:30]
    amostra = "\n".join(linhas)
    if re.search(r"\d,\d", amostra):
        return ","
    return "."


def _yyyymm_em_caminho(path: Path, base: Path | None = None) -> str | None:
    if base is not None:
        try:
            partes = list(path.relative_to(base).parts)
        except ValueError:
            partes = list(path.parts)
    else:
        partes = [path.name]
    achados = []
    for parte in partes:
        achados.extend(_YYYYMM.findall(parte))
    if not achados:
        return None
    return max(ano + mes for ano, mes in achados)


def _rank_bolsa(path: Path, base: Path) -> tuple:
    referencia = _yyyymm_em_caminho(path, base)
    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0
    if referencia:
        return (1, int(referencia), mtime)
    return (0, 0, mtime)


def _listar_bolsa_local(base: Path) -> list[Path]:
    """Varre base/**/* sem descer em raw/ (lá está a base de BUs)."""
    encontrados = []
    saida = (DATA_SOCIO / _SAIDA_BOLSA).resolve()

    def _aviso(err):
        print(f"Aviso: falha ao listar diretório ({err})")

    for root, dirs, files in os.walk(base, onerror=_aviso):
        dirs[:] = [d for d in dirs if d.lower() != "raw"]
        for nome in files:
            lower = nome.lower()
            if not lower.endswith((".zip", ".csv")):
                continue
            if "bolsa" not in lower and "novobolsa" not in lower:
                continue
            caminho = Path(root) / nome
            try:
                if caminho.resolve() == saida:
                    continue
            except OSError:
                pass
            encontrados.append(caminho)
    return encontrados


def _eh_zip(path: Path) -> bool:
    try:
        return zipfile.is_zipfile(path)
    except OSError:
        return False


def _escolher_membro_csv(zf: zipfile.ZipFile) -> zipfile.ZipInfo:
    infos = [i for i in zf.infolist() if not i.is_dir() and not i.filename.startswith("__MACOSX")]
    if not infos:
        raise ValueError("zip sem arquivos")
    csvs = [i for i in infos if i.filename.lower().endswith(".csv")]
    pool = csvs or infos
    com_bolsa = [i for i in pool if "bolsa" in i.filename.lower()]
    pool = com_bolsa or pool
    return max(pool, key=lambda i: i.file_size)


def _abrir_csv_binario(path: Path):
    """Devolve (zip|None, arquivo binário, meta). O chamador fecha os dois."""
    if _eh_zip(path):
        zf = zipfile.ZipFile(path)
        try:
            membro = _escolher_membro_csv(zf)
            bruto = zf.open(membro)
        except Exception:
            zf.close()
            raise
        return zf, bruto, {"membros": list(zf.infolist()), "membro": membro.filename}
    return None, open(path, "rb"), {"membros": None, "membro": None}


def _fechar_csv_binario(zf, bruto) -> None:
    try:
        if bruto is not None:
            bruto.close()
    finally:
        if zf is not None:
            zf.close()


def _ler_amostra(path: Path) -> tuple[bytes, dict]:
    zf, bruto, meta = _abrir_csv_binario(path)
    try:
        meta["amostra"] = bruto.read(BOLSA_AMOSTRA)
    finally:
        _fechar_csv_binario(zf, bruto)
    return meta["amostra"], meta


def _log_membros_zip(meta: dict) -> None:
    membros = meta.get("membros")
    if not membros:
        return
    usado = meta.get("membro")
    print(f"Membros do zip ({len(membros)}):")
    for info in membros:
        if info.is_dir():
            tamanho = "dir"
        else:
            tamanho = _fmt_tamanho(info.file_size)
        marca = " <-- csv usado" if info.filename == usado else ""
        print(f"  - {info.filename} ({tamanho}){marca}")


def _mapear_colunas(colunas: list[str]) -> dict:
    normas = {c: _sem_acento(c) for c in colunas}

    def achar(pred):
        for coluna, norma in normas.items():
            if pred(norma):
                return coluna
        return None

    codigo_siafi = achar(lambda n: "siafi" in n and ("munic" in n or "codigo" in n or n.startswith("cod ")))
    codigo_ibge = achar(lambda n: "ibge" in n and ("munic" in n or "codigo" in n or n.startswith("cod ")))
    if codigo_siafi:
        codigo, tipo = codigo_siafi, "siafi"
    elif codigo_ibge:
        codigo, tipo = codigo_ibge, "ibge"
    else:
        codigo = achar(lambda n: ("codigo" in n or n.startswith("cod ")) and "munic" in n)
        tipo = "municipio" if codigo else None
        if codigo is None:
            codigo = achar(lambda n: n in {"codigo", "cd municipio", "cd mun"})
            tipo = "municipio" if codigo else None

    nome = achar(lambda n: "nome" in n and "munic" in n and "favorecido" not in n)
    uf = achar(lambda n: n in {"uf", "sg uf", "sigla uf"})
    valor = achar(lambda n: "valor" in n and "parcela" in n)
    if valor is None:
        valor = achar(lambda n: n == "valor" or (n.startswith("valor ") and "favorecido" not in n))
    mes_ref = achar(lambda n: "mes" in n and "referenc" in n)
    mes_comp = achar(lambda n: "mes" in n and "competenc" in n)
    if mes_ref is None and mes_comp is not None:
        mes_ref = mes_comp
        mes_comp = None
    if codigo is None and nome is not None:
        codigo = nome
        tipo = "nome"
    return {
        "codigo": codigo,
        "tipo": tipo,
        "nome": nome,
        "uf": uf,
        "valor": valor,
        "mes_ref": mes_ref,
        "mes_comp": mes_comp,
    }


def _para_numero(serie: pd.Series, decimal: str) -> pd.Series:
    s = serie.fillna("").astype(str).str.strip()
    s = s.str.replace("\xa0", "", regex=False).str.replace(" ", "", regex=False)
    s = s.str.replace("R$", "", regex=False)
    if decimal == ",":
        s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    else:
        s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce").fillna(0.0)


def _limpar_codigo(serie: pd.Series) -> pd.Series:
    s = serie.fillna("").astype(str).str.strip()
    return s.str.replace(r"\.0$", "", regex=True)


def _agregar_chunk(chunk: pd.DataFrame, spec: dict, decimal: str) -> pd.DataFrame:
    df = pd.DataFrame(index=chunk.index)
    df["codigo_municipio"] = _limpar_codigo(chunk[spec["codigo"]])
    if spec["nome"] and spec["nome"] != spec["codigo"]:
        df["nome_municipio"] = chunk[spec["nome"]].fillna("").astype(str).str.strip()
    elif spec["tipo"] == "nome":
        df["nome_municipio"] = df["codigo_municipio"]
    if spec["uf"]:
        df["uf"] = chunk[spec["uf"]].fillna("").astype(str).str.strip().str.upper()
    if spec["mes_ref"]:
        df["mes_referencia"] = chunk[spec["mes_ref"]].fillna("").astype(str).str.strip()
    if spec["mes_comp"]:
        df["mes_competencia"] = chunk[spec["mes_comp"]].fillna("").astype(str).str.strip()
    if spec["valor"]:
        df["valor_parcela"] = _para_numero(chunk[spec["valor"]], decimal)
    df["n_beneficiarios"] = 1
    return _reduzir(df)


def _reduzir(df: pd.DataFrame) -> pd.DataFrame:
    grupo = [c for c in ("codigo_municipio", "uf", "mes_referencia", "mes_competencia") if c in df.columns]
    agregacoes = {"n_beneficiarios": "sum"}
    if "valor_parcela" in df.columns:
        agregacoes["valor_parcela"] = "sum"
    if "nome_municipio" in df.columns:
        agregacoes["nome_municipio"] = "first"
    return df.groupby(grupo, dropna=False, as_index=False).agg(agregacoes)


def _colunas_do_cabecalho(cabecalho: str, sep: str) -> list[str]:
    if not cabecalho.strip():
        return []
    return next(csv.reader([cabecalho], delimiter=sep))


def _iter_chunks(path: Path, encoding: str, sep: str, usecols: list[str] | None):
    def abrir(cols):
        zf, bruto, _meta = _abrir_csv_binario(path)
        try:
            leitor = pd.read_csv(
                bruto,
                sep=sep,
                encoding=encoding,
                dtype=str,
                chunksize=BOLSA_CHUNK,
                usecols=cols,
                on_bad_lines="warn",
            )
        except Exception:
            _fechar_csv_binario(zf, bruto)
            raise
        return zf, bruto, leitor

    zf, bruto, leitor = abrir(usecols)
    try:
        try:
            iterador = iter(leitor)
            primeiro = next(iterador)
        except StopIteration:
            return
        except ValueError:
            if not usecols:
                raise
            print("Aviso: colunas pedidas não bateram com o cabeçalho; lendo todas as colunas")
            _fechar_csv_binario(zf, bruto)
            zf, bruto, leitor = abrir(None)
            iterador = iter(leitor)
            try:
                primeiro = next(iterador)
            except StopIteration:
                return
        yield primeiro
        yield from iterador
    finally:
        _fechar_csv_binario(zf, bruto)


def _agregar_arquivo(path: Path, encoding: str, sep: str, decimal: str, spec: dict, mes_hint: str | None, min_linhas: int) -> Path | None:
    necessarias = [spec[k] for k in ("codigo", "nome", "uf", "valor", "mes_ref", "mes_comp") if spec.get(k)]
    # preserva a ordem e remove repetidos (codigo pode ser o próprio nome)
    usecols = list(dict.fromkeys(necessarias))
    partes = []
    total = 0
    for chunk in _iter_chunks(path, encoding, sep, usecols):
        chunk.columns = [str(c).strip() for c in chunk.columns]
        total += len(chunk)
        partes.append(_agregar_chunk(chunk, spec, decimal))
        print(f"  Linhas lidas: {total}")
    if total < min_linhas:
        print("    Arquivo muito pequeno, ignorando")
        return None
    out = _reduzir(pd.concat(partes, ignore_index=True))
    if spec["valor"] and "valor_parcela" in out.columns:
        out["valor_parcela"] = out["valor_parcela"].round(2)
        out["bolsa_familia_valor_ou_qtd"] = out["valor_parcela"]
    elif "valor_parcela" not in out.columns:
        print("AVISO: coluna de valor da parcela não encontrada; só a contagem de beneficiários foi agregada")
    out["tipo_codigo"] = spec["tipo"] or ""
    if "mes_referencia" not in out.columns and mes_hint:
        out["mes_referencia"] = str(mes_hint)
    elif "mes_referencia" in out.columns and mes_hint:
        vazio = out["mes_referencia"].astype(str).str.strip() == ""
        out.loc[vazio, "mes_referencia"] = str(mes_hint)
    out["n_beneficiarios"] = out["n_beneficiarios"].astype(int)
    ordem = [
        "codigo_municipio",
        "tipo_codigo",
        "nome_municipio",
        "uf",
        "mes_referencia",
        "mes_competencia",
        "n_beneficiarios",
        "valor_parcela",
        "bolsa_familia_valor_ou_qtd",
    ]
    out = out[[c for c in ordem if c in out.columns]]
    sort_cols = [c for c in ("uf", "codigo_municipio", "mes_referencia") if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    DATA_SOCIO.mkdir(parents=True, exist_ok=True)
    out_path = DATA_SOCIO / _SAIDA_BOLSA
    out.to_csv(out_path, index=False, encoding="utf-8", float_format="%.2f")
    beneficiarios = int(out["n_beneficiarios"].sum())
    if "valor_parcela" in out.columns:
        soma = float(out["valor_parcela"].sum())
        print(f"Agregado: {len(out)} registros (município/mês), {beneficiarios} beneficiários, soma das parcelas={soma:.2f}")
    else:
        print(f"Agregado: {len(out)} registros (município/mês), {beneficiarios} beneficiários")
    print(f"Salvo em {out_path} ({len(out)} registros)")
    return out_path


def processar_arquivo_bolsa(path: Path, mes_hint: str | None = None, min_linhas: int = 1) -> Path | None:
    """Lê um ZIP/CSV do Novo Bolsa Família e grava o agregado municipal."""
    tamanho = path.stat().st_size
    print(f"Lendo Bolsa Família: {path} ({_fmt_tamanho(tamanho)})")
    amostra, meta = _ler_amostra(path)
    _log_membros_zip(meta)
    if _parece_html(amostra):
        raise ValueError("conteúdo HTML, não CSV")
    if not amostra:
        raise ValueError("arquivo vazio")
    ultimo_erro: Exception | None = None
    for encoding in _ordem_encodings(amostra):
        try:
            texto = amostra.decode(encoding)
        except UnicodeDecodeError as exc:
            print(f"Encoding {encoding} não decodifica a amostra ({exc})")
            ultimo_erro = exc
            continue
        linhas = texto.splitlines()
        if not linhas or not linhas[0].strip():
            raise ValueError("CSV sem cabeçalho")
        sep = _detectar_sep(linhas[0])
        decimal = _detectar_decimal(texto, sep)
        print(f"Encoding: {encoding}")
        print(f"Separador: {sep!r}")
        print(f"Decimal: {decimal!r}")
        print("Primeiras linhas do CSV:")
        for i, linha in enumerate(linhas[:3], 1):
            print(f"  {i}: {linha[:1000]}")
        colunas = [c.strip() for c in _colunas_do_cabecalho(linhas[0], sep)]
        spec = _mapear_colunas(colunas)
        print(f"Colunas encontradas: {colunas}")
        print(
            "Colunas usadas: "
            f"codigo={spec['codigo']} ({spec['tipo']}), nome={spec['nome']}, uf={spec['uf']}, "
            f"valor={spec['valor']}, mes_referencia={spec['mes_ref']}, mes_competencia={spec['mes_comp']}"
        )
        if spec["codigo"] is None:
            raise ValueError(f"Não achei coluna de município em {colunas}")
        if spec["mes_ref"] and spec["mes_comp"] is None and "competenc" in _sem_acento(spec["mes_ref"]):
            print("Mês referência ausente; usando mês competência como mes_referencia")
        try:
            return _agregar_arquivo(path, encoding, sep, decimal, spec, mes_hint, min_linhas)
        except UnicodeError as exc:
            print(f"Falha com encoding {encoding}: {exc}")
            ultimo_erro = exc
            continue
    if ultimo_erro:
        raise ultimo_erro
    raise ValueError("não foi possível decodificar o CSV")


def baixar_bolsa_local():
    """Usa o Novo Bolsa Família já baixado em SOCIO_LOCAL_DIR, se existir."""
    base = Path(os.environ.get("SOCIO_LOCAL_DIR", "/home/alexa/eleicoes-data"))
    print(f"Procurando Bolsa Família em {base} (SOCIO_LOCAL_DIR)")
    if not base.is_dir():
        print(f"Diretório local inexistente: {base}")
        return None
    pasta_raw = base / "raw"
    if pasta_raw.exists():
        print(f"Ignorando {pasta_raw} (base de BUs, não entra na busca)")
    candidatos = _listar_bolsa_local(base)
    if not candidatos:
        print(f"Nenhum arquivo local (.zip/.csv com 'bolsa' no nome) em {base}")
        return None
    ordenados = sorted(candidatos, key=lambda p: _rank_bolsa(p, base), reverse=True)
    referencia = _yyyymm_em_caminho(ordenados[0], base)
    criterio = f"AAAAMM {referencia}" if referencia else "mtime"
    print(f"Candidatos: {len(ordenados)}. Mais recente por {criterio}: {ordenados[0]} ({_fmt_tamanho(ordenados[0].stat().st_size)})")
    for caminho in ordenados:
        try:
            saida = processar_arquivo_bolsa(caminho, mes_hint=_yyyymm_em_caminho(caminho, base))
        except Exception as exc:
            print(f"Falha ao ler {caminho}: {exc}")
            continue
        if saida is not None:
            print(f"Bolsa Família obtido do arquivo local: {caminho}")
            return saida
    print("Nenhum arquivo local de Bolsa Família pôde ser lido.")
    return None


def _corpo_parece_zip(conteudo: bytes) -> bool:
    return conteudo.startswith((b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08"))


def baixar_bolsa_remoto():
    """Tenta o Portal da Transparência. HTML de erro não vai para o pandas."""
    meses = gerar_meses_recentes(12)
    print(f"Tentando Bolsa Família nos meses: {meses[:6]}...")

    for yyyymm in meses:
        url = f"https://portaldatransparencia.gov.br/download-de-dados/novo-bolsa-familia/{yyyymm}"
        print(f"  Tentando {yyyymm}...")
        tmp_path = None
        try:
            with requests.get(url, timeout=90, stream=True) as r:
                if r.status_code != 200:
                    print(f"    HTTP {r.status_code}")
                    continue
                primeiro = b""
                iterator = r.iter_content(1024 * 256)
                for bloco in iterator:
                    if bloco:
                        primeiro = bloco
                        break
                tipo = r.headers.get("Content-Type", "")
                if _parece_html(primeiro, tipo):
                    print(f"    Resposta HTML (não é CSV); ignorando {yyyymm}")
                    continue
                sufixo = ".zip" if _corpo_parece_zip(primeiro) else ".csv"
                fd, tmp_path = tempfile.mkstemp(prefix=f"bolsa_{yyyymm}_", suffix=sufixo)
                with os.fdopen(fd, "wb") as tmp:
                    tmp.write(primeiro)
                    for bloco in iterator:
                        if bloco:
                            tmp.write(bloco)
            saida = processar_arquivo_bolsa(Path(tmp_path), mes_hint=yyyymm, min_linhas=10)
            if saida is not None:
                print(f"  Sucesso com {yyyymm}!")
                return saida
        except Exception as e:
            texto = str(e)
            if "HTML" in texto or "não é CSV" in texto:
                print(f"    {texto}")
            else:
                print(f"    Erro: {e}")
            continue
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)

    print("Nenhum mês recente funcionou. Verifique manualmente em https://portaldatransparencia.gov.br/download-de-dados/novo-bolsa-familia")
    return None


def baixar_bolsa_familia():
    """Prefere o arquivo local. O portal só é tentado se nenhum arquivo local servir."""
    try:
        local = baixar_bolsa_local()
    except Exception as exc:
        print(f"Erro ao procurar Bolsa Família local: {exc}")
        local = None
    if local is not None:
        return local
    return baixar_bolsa_remoto()

def baixar_ivs():
    print("IVS: base municipal completa disponível é de 2010.")
    print("Acesse https://ivs.ipea.gov.br/ ou dados.gov.br para download manual do XLSX municipal.")
    print("Para análise municipal, use o arquivo idh_municipios.csv ou baixe manualmente.")

def main():
    print("Baixando bases socioeconômicas...")
    try:
        baixar_idh()
    except Exception as e:
        print(f"Erro ao baixar IDH: {e}")
    try:
        baixar_transferencias()
    except Exception as e:
        print(f"Erro ao baixar transferências: {e}")
    try:
        baixar_bolsa_familia()
    except Exception as e:
        print(f"Erro ao baixar Bolsa Família: {e}")
    baixar_ivs()
    print("Concluído. Arquivos em data/socioeconomico/")

if __name__ == "__main__":
    main()
