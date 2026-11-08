#!/usr/bin/env python3
"""
Processa Boletins de Urna baixados (zips em data/raw/),
resume por seção e grava saídas úteis em resultados/.
Não grava arquivos crus em resultados/.
"""

import zipfile
from pathlib import Path
import pandas as pd

RAW = Path("data/raw")
RESULTS = Path("resultados")
RESULTS.mkdir(exist_ok=True)

def load_bu_zip(zip_path: Path) -> pd.DataFrame:
    """Carrega o CSV principal de dentro do zip do BU."""
    with zipfile.ZipFile(zip_path) as z:
        # Pega o primeiro arquivo .csv ou .txt grande
        csv_names = [n for n in z.namelist() if n.lower().endswith(('.csv', '.txt'))]
        if not csv_names:
            raise ValueError(f"Nenhum CSV encontrado em {zip_path.name}")
        # Usa o maior (geralmente o principal)
        csv_name = max(csv_names, key=lambda n: z.getinfo(n).file_size)
        with z.open(csv_name) as f:
            # TSE costuma usar ; e encoding latin-1 ou utf-8
            for enc in ('utf-8', 'latin-1', 'cp1252'):
                try:
                    f.seek(0)
                    df = pd.read_csv(f, sep=';', encoding=enc, low_memory=False, dtype=str)
                    return df
                except Exception:
                    continue
            raise ValueError(f"Não foi possível ler {csv_name} em {zip_path.name}")

def summarize_df(df: pd.DataFrame, label: str) -> dict:
    """Resumo básico de um BU."""
    summary = {
        "label": label,
        "linhas": len(df),
        "colunas": list(df.columns[:20]),  # primeiras colunas para inspeção
        "n_colunas": len(df.columns),
    }
    # Tenta identificar colunas típicas
    for col in ['SG_UF', 'NR_ZONA', 'NR_SECAO', 'QT_COMPARECIMENTO', 'QT_APTOS']:
        if col in df.columns:
            summary[f"tem_{col}"] = True
            if col in ['QT_COMPARECIMENTO', 'QT_APTOS']:
                try:
                    summary[f"soma_{col}"] = pd.to_numeric(df[col], errors='coerce').sum()
                except Exception:
                    pass
    return summary

def main():
    zips = sorted(RAW.glob("bweb_*.zip"))
    if not zips:
        print("Nenhum zip encontrado em data/raw/. Rode o download primeiro.")
        return

    reports = []
    dfs = {}

    for zp in zips:
        print(f"Processando {zp.name}...")
        try:
            df = load_bu_zip(zp)
            label = zp.stem
            dfs[label] = df
            summary = summarize_df(df, label)
            reports.append(summary)
            print(f"  -> {len(df)} linhas, {len(df.columns)} colunas")
        except Exception as e:
            print(f"  ERRO em {zp.name}: {e}")
            reports.append({"label": zp.name, "erro": str(e)})

    # Grava resumo geral
    out_txt = RESULTS / "resumo_bus.txt"
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("Resumo dos Boletins de Urna processados\n")
        f.write("=" * 50 + "\n\n")
        for r in reports:
            f.write(str(r) + "\n\n")

    # Se houver T1 e T2 da mesma UF, tenta juntar por seção
    # (exemplo simples para AC)
    t1_keys = [k for k in dfs if "1t" in k]
    t2_keys = [k for k in dfs if "2t" in k]

    join_info = []
    for t1k in t1_keys:
        uf = t1k.split("_")[2] if "_" in t1k else "?"
        t2k = next((k for k in t2_keys if uf in k), None)
        if t2k and t1k in dfs and t2k in dfs:
            df1 = dfs[t1k]
            df2 = dfs[t2k]
            # Tenta chaves comuns de seção
            keys = [c for c in ['SG_UF', 'CD_MUNICIPIO', 'NR_ZONA', 'NR_SECAO'] if c in df1.columns and c in df2.columns]
            if keys:
                merged = df1.merge(df2, on=keys, how='inner', suffixes=('_t1', '_t2'))
                join_info.append({
                    "uf": uf,
                    "secoes_t1": len(df1),
                    "secoes_t2": len(df2),
                    "secoes_join": len(merged),
                    "chaves": keys
                })
                # Salva uma amostra pequena (não o arquivo cru)
                sample = merged.head(5)
                sample.to_csv(RESULTS / f"amostra_join_{uf}.csv", index=False, sep=';')
                print(f"Join {uf}: {len(merged)} seções comuns")

    if join_info:
        with open(RESULTS / "joins_resumo.txt", "w", encoding="utf-8") as f:
            f.write("Resumo de junções T1 x T2 por seção\n")
            f.write("=" * 40 + "\n")
            for j in join_info:
                f.write(str(j) + "\n")

    print(f"\nResultados gravados em {RESULTS.resolve()}")
    print("Arquivos crus permanecem apenas em data/raw/")

if __name__ == "__main__":
    main()
