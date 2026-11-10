#!/usr/bin/env python3
"""
Baixa os arquivos de log da urna (2º turno) do TSE.
Os ZIPs contêm .logjez, .rdv, .bu, etc.
"""

from pathlib import Path
import urllib.request
import sys

LOGS_DIR = Path("/home/alexa/eleicoes-data/logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes2022/arqurnatot/bu_imgbu_logjez_rdv_vscmr_2022_2t_{uf}.zip"

# Começa com UFs menores / de interesse
UFS = ["AC", "AL", "AP", "RR", "SE", "TO", "DF", "ES", "MS", "MT"]

def download_uf(uf: str):
    url = BASE_URL.format(uf=uf)
    dest = LOGS_DIR / f"logs_2t_{uf}.zip"
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"  {uf}: já existe ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    print(f"  Baixando {uf}...")
    try:
        urllib.request.urlretrieve(url, dest)
        print(f"  {uf}: ok ({dest.stat().st_size / 1e6:.1f} MB)")
    except Exception as e:
        print(f"  {uf}: erro - {e}")

def main():
    print(f"Baixando logs 2º turno para {len(UFS)} UFs...")
    for uf in UFS:
        download_uf(uf)
    print(f"Logs em {LOGS_DIR}")

if __name__ == "__main__":
    main()
