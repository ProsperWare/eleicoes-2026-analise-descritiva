#!/usr/bin/env python3
"""
Baixa e prepara bases oficiais para cruzamento socioeconômico:
- IDHM municipal (base Atlas Brasil 2010, via GitHub público)
- Transferências constitucionais mensais (Tesouro Transparente)

Salva em data/socioeconomico/
"""

from pathlib import Path
import requests
import pandas as pd
import io

DATA_SOCIO = Path("data/socioeconomico")
DATA_SOCIO.mkdir(parents=True, exist_ok=True)

def baixar_idh():
    # Base pública do Atlas Brasil (IDHM 2010 e indicadores) — GitHub mauriciocramos/IDHM
    url = "https://raw.githubusercontent.com/mauriciocramos/IDHM/master/municipal.csv"
    print(f"Baixando IDHM de {url}")
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=";", encoding="utf-8", low_memory=False)
    # Mantém colunas úteis
    cols_uteis = [c for c in df.columns if c.lower() in ("cd_mun", "municipio", "uf", "idhm", "idhm_e", "idhm_l", "idhm_r", "pop")]
    if not cols_uteis:
        cols_uteis = df.columns[:8].tolist()  # fallback
    df_out = df[cols_uteis].copy()
    out_path = DATA_SOCIO / "idh_municipios.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8")
    print(f"IDHM salvo em {out_path} ({len(df_out)} municípios)")
    return out_path

def baixar_transferencias():
    # CSV mensal mais recente do Tesouro Transparente (exemplo 202606)
    url = "https://www.tesourotransparente.gov.br/ckan/dataset/af4e7c47-2132-4d9a-bd7c-34e28a210b03/resource/315e91df-109d-41c2-b8c6-820eb3ffe0cd/download/transferenciamensalmunicipios202606.csv.csv"
    print(f"Baixando transferências de {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep=";", encoding="latin1", low_memory=False)
    # Agrega por município (soma do período)
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
    print("Concluído. Arquivos em data/socioeconomico/")

if __name__ == "__main__":
    main()
