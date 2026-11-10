#!/usr/bin/env python3
"""
Cruza residual agregado por município com bases oficiais:
- IDH municipal (Atlas Brasil / PNUD)
- Recebimento líquido de recursos federais (Portal da Transparência)

Fontes:
- IDHM: https://www.atlasbrasil.org.br/ (download CSV municipal)
- Transferências: https://portaldatransparencia.gov.br/transferencias/consulta
  ou dados.gov.br (Transferências Constitucionais para Municípios)

Uso:
  # Coloque os CSVs em data/socioeconomico/
  # idh_municipios.csv e transferencias_municipios.csv
  python scripts/cruzar_socioeconomico.py --ano 2026
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np

PROCESSED = Path("/home/alexa/eleicoes-data/processed")
DATA_SOCIO = Path("data/socioeconomico")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)
DATA_SOCIO.mkdir(parents=True, exist_ok=True)

def load_residual_by_municipio(ano: str) -> pd.DataFrame:
    """Agrega residual médio por município a partir dos joins processados."""
    join_files = sorted(PROCESSED.glob(f"join_*_{ano}.parquet")) or sorted(PROCESSED.glob("join_*.parquet"))
    if not join_files:
        print("Nenhum join encontrado. Rode o preprocess primeiro.")
        return pd.DataFrame()

    frames = []
    for jf in join_files:
        try:
            df = pd.read_parquet(jf)
        except Exception:
            continue
        if df.empty:
            continue

        # Tenta identificar colunas de residual ou recalcula de forma simples
        # Aqui assumimos que o residual já pode estar ou que usamos PCT de um candidato principal
        # Para simplificar: média de um residual proxy se existir
        muni_col = next((c for c in ["CD_MUNICIPIO", "NM_MUNICIPIO", "municipio"] if c in df.columns), None)
        if muni_col is None:
            continue

        # Proxy residual: se houver coluna residual, usa; senão pula
        if "residual" in df.columns:
            agg = df.groupby(muni_col)["residual"].agg(["mean", "std", "count"]).reset_index()
            agg.columns = [muni_col, "residual_mean", "residual_std", "n_secoes"]
            frames.append(agg)

    if not frames:
        print("Não foi possível agregar residual por município. Verifique colunas nos joins.")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ano", default="2026")
    args = parser.parse_args()

    print(f"Cruzamento socioeconômico para {args.ano}")

    res_muni = load_residual_by_municipio(args.ano)
    if res_muni.empty:
        print("Sem dados de residual agregado.")
        return

    # IDH
    idh_path = DATA_SOCIO / "idh_municipios.csv"
    if idh_path.exists():
        idh = pd.read_csv(idh_path, sep=None, engine="python", encoding="utf-8")
        # Normaliza nomes de coluna comuns
        idh.columns = [c.strip().lower().replace(" ", "_") for c in idh.columns]
        print(f"IDH carregado: {len(idh)} municípios")
    else:
        print(f"AVISO: {idh_path} não encontrado. Baixe do Atlas Brasil e salve como idh_municipios.csv")
        idh = pd.DataFrame()

    # Transferências
    trans_path = DATA_SOCIO / "transferencias_municipios.csv"
    if trans_path.exists():
        trans = pd.read_csv(trans_path, sep=None, engine="python", encoding="utf-8")
        trans.columns = [c.strip().lower().replace(" ", "_") for c in trans.columns]
        print(f"Transferências carregadas: {len(trans)} registros")
    else:
        print(f"AVISO: {trans_path} não encontrado. Exporte do Portal da Transparência (consulta por município, tipo favorecido = Administração Pública Municipal)")
        trans = pd.DataFrame()

    # Join básico (por nome de município — idealmente use código IBGE)
    out = res_muni.copy()
    if not idh.empty:
        # Tenta join por nome ou código
        muni_key = "cd_municipio" if "cd_municipio" in out.columns else out.columns[0]
        idh_key = next((c for c in idh.columns if "munic" in c or "codigo" in c), idh.columns[0])
        out = out.merge(idh, left_on=muni_key, right_on=idh_key, how="left")

    if not trans.empty:
        muni_key = "cd_municipio" if "cd_municipio" in out.columns else out.columns[0]
        trans_key = next((c for c in trans.columns if "munic" in c or "codigo" in c or "favorecido" in c), trans.columns[0])
        out = out.merge(trans, left_on=muni_key, right_on=trans_key, how="left")

    # Correlações simples
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    if "residual_mean" in numeric_cols:
        corrs = out[numeric_cols].corr()["residual_mean"].sort_values(ascending=False)
        with open(RESULTS / f"correlacao_socio_{args.ano}.txt", "w", encoding="utf-8") as f:
            f.write(f"Correlações com residual_mean ({args.ano})\n")
            f.write("=" * 50 + "\n")
            f.write(corrs.to_string())
            f.write(f"\n\nN municípios: {len(out)}\n")
        print(corrs.head(10))

    out.to_csv(RESULTS / f"residual_socio_{args.ano}.csv", index=False)
    print(f"Resultado salvo em resultados/residual_socio_{args.ano}.csv")
    print("\nPróximos passos:")
    print("1. Baixe IDH municipal do Atlas Brasil e salve em data/socioeconomico/idh_municipios.csv")
    print("2. Exporte transferências do Portal da Transparência (filtro: município + Administração Pública Municipal) e salve em data/socioeconomico/transferencias_municipios.csv")
    print("3. Ajuste as chaves de join se os nomes das colunas forem diferentes")

if __name__ == "__main__":
    main()
