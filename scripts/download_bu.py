#!/usr/bin/env python3
"""
Download Boletins de Urna (BU) oficiais das Eleições 2026 do TSE.
Fonte: https://dadosabertos.tse.jus.br/dataset/resultados-2026-boletim-de-urna

Arquivos baixados ficam em data/raw/. Não coloque arquivos crus em resultados/.
"""

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

BASE_URL = "https://cdn.tse.jus.br/estatistica/sead/eleicoes/eleicoes2026/buweb"
UFS = [
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA",
    "MG", "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN",
    "RO", "RR", "RS", "SC", "SE", "SP", "TO", "ZZ"  # ZZ = exterior
]

def sha512_file(path: Path) -> str:
    h = hashlib.sha512()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)

def verify(zip_path: Path, sha_path: Path) -> bool:
    if not sha_path.exists():
        print(f"Aviso: hash não encontrado para {zip_path.name}")
        return False
    expected = sha_path.read_text().strip().split()[0]
    actual = sha512_file(zip_path)
    ok = actual.lower() == expected.lower()
    print(f"Hash {'OK' if ok else 'FALHOU'} para {zip_path.name}")
    return ok

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ufs", nargs="*", default=[], help="UFs específicas (ex: AC SP). Padrão: nenhuma (use --all)")
    parser.add_argument("--all", action="store_true", help="Baixar todas as UFs")
    parser.add_argument("--raw-dir", default="data/raw", help="Pasta para arquivos crus")
    args = parser.parse_args()

    ufs = UFS if args.all else [u.upper() for u in args.ufs]
    if not ufs:
        print("Nenhuma UF especificada. Use --all ou --ufs AC SP ...")
        return

    raw = Path(args.raw_dir)
    for uf in ufs:
        for turn, stamp in [("1t", "051020261403"), ("2t", "261020260742")]:
            name = f"bweb_{turn}_{uf}_{stamp}.zip"
            url = f"{BASE_URL}/{name}"
            zip_path = raw / name
            sha_path = raw / f"{name}.sha512"
            if not zip_path.exists():
                download(url, zip_path)
                download(f"{url}.sha512", sha_path)
            verify(zip_path, sha_path)

    print("Download concluído. Arquivos crus em", raw.resolve())
    print("Resultados processados devem ir para a pasta resultados/ (sem arquivos crus).")

if __name__ == "__main__":
    main()
