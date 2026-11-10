#!/usr/bin/env python3
"""
Baixa e prepara bases oficiais para cruzamento socioeconômico:
- IDHM municipal (base Atlas Brasil — último censo completo 2010)
- Transferências constitucionais mensais (Tesouro Transparente)
- Bolsa Família / Novo Bolsa Família (Portal da Transparência)
- Nota sobre IVS (Ipea)

Nota sobre IDHM:
  O último IDHM municipal completo e público em CSV é o de 2010 (Censo).
  O Radar IDHM 2024 (2012-2024, PNAD Contínua) cobre Brasil, UFs e Regiões
  Metropolitanas, mas NÃO o nível municipal individual.

Salva em data/socioeconomico/
"""

from pathlib import Path
import requests
import pandas as pd
import io

DATA_SOCIO = Path("data/socioeconomico")
DATA_SOCIO.mkdir(parents=True, exist_ok=True)

def baixar_idh():
    # Base pública do Atlas Brasil (IDHM municipal 2010) — GitHub mauriciocramos/IDHM
    url = "https://raw.githubusercontent.com/mauriciocramos/IDHM/master/municipal.csv"
    print(f"Baixando IDHM municipal (base 2010) de {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=";", encoding="utf-8", low_memory=False)
    cols_uteis = [c for c in df.columns if c.lower() in ("cd_mun", "municipio", "uf", "idhm", "idhm_e", "idhm_l", "idhm_r", "pop")]
    if not cols_uteis:
        cols_uteis = df.columns[:8].tolist()
    df_out = df[cols_uteis].copy()
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
    if "Valor" in df.columns or "valor" in df.columns:
        valor_col = "Valor" if "Valor" in df.columns else "valor"
        muni_col = next((c for c in df.columns if "munic" in c.lower() or "nome" in c.lower()), df.columns[0])
        agg = df.groupby(muni_col)[valor_col].sum().reset_index()
        agg.columns = [muni_col, "transferencia_liquida"]
    else:
        agg = df
    out_path = DATA_SOCIO / "transferencias_municipios.csv"
    agg.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Transferências salvas em {out_path} ({len(agg)} registros)")
    return out_path

def baixar_bolsa_familia():
    # Novo Bolsa Família — Portal da Transparência (exemplo mês recente)
    # Padrão: https://portaldatransparencia.gov.br/download-de-dados/novo-bolsa-familia/YYYYMM
    url = "https://portaldatransparencia.gov.br/download-de-dados/novo-bolsa-familia/202506"
    print(f"Baixando Bolsa Família de {url}")
    try:
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        # O portal retorna CSV direto ou zip; aqui assumimos CSV
        df = pd.read_csv(io.StringIO(r.text), sep=";", encoding="latin1", low_memory=False)
        # Agrega por município se possível
        muni_col = next((c for c in df.columns if "munic" in c.lower() or "ibge" in c.lower() or "codigo" in c.lower()), None)
        valor_col = next((c for c in df.columns if "valor" in c.lower() or "qtd" in c.lower()), None)
        if muni_col and valor_col:
            agg = df.groupby(muni_col)[valor_col].sum().reset_index()
            agg.columns = [muni_col, "bolsa_familia_valor_ou_qtd"]
        else:
            agg = df
        out_path = DATA_SOCIO / "bolsa_familia_municipios.csv"
        agg.to_csv(out_path, index=False, encoding="utf-8")
        print(f"Bolsa Família salva em {out_path} ({len(agg)} registros)")
        return out_path
    except Exception as e:
        print(f"Erro ao baixar Bolsa Família: {e}")
        print("Dica: verifique o mês disponível em https://portaldatransparencia.gov.br/download-de-dados/novo-bolsa-familia")
        return None

def baixar_ivs():
    # IVS municipal completo é principalmente base 2010 (Censo).
    # Dados mais recentes (PNAD) estão em nível de UF no Ipeadata.
    print("IVS: base municipal completa disponível é de 2010.")
    print("Acesse https://ivs.ipea.gov.br/ ou dados.gov.br para download manual do XLSX municipal.")
    print("Para análise municipal, use o arquivo idh_municipios.csv (que já traz indicadores correlatos) ou baixe manualmente.")

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
