#!/usr/bin/env python3
"""
Download Boletins de Urna (BU) oficiais das Eleicoes 2014 do TSE.
Fonte: https://dadosabertos.tse.jus.br/dataset/resultados-2014-boletim-de-urna

O endpoint devolve um ZIP contendo o CSV. Extrai o CSV e salva em data/raw/2014/.
"""

from __future__ import annotations

import json
import re
import tempfile
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://dadosabertos.tse.jus.br/api/3/action/package_show?id=resultados-2014-boletim-de-urna"
UFS = {
    "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG", "MS", "MT",
    "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR", "RS", "SC", "SE", "SP", "TO", "ZZ",
}


def is_valid_csv(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 1000:
        return False
    try:
        with open(path, "rb") as f:
            head = f.read(200)
            if head.startswith(b"PK"):
                return False
            text = head.decode("utf-8", errors="ignore").lower()
            if "<html" in text or "<!doctype" in text:
                return False
            return ";" in text and any(x in text for x in ("cd_cargo", "qt_votos", "nr_secao", "sg_uf"))
    except Exception:
        return False


def load_resources() -> dict[str, dict[str, str]]:
    with urllib.request.urlopen(DATASET_URL, timeout=30) as resp:
        payload = json.load(resp)
    if not payload.get("success"):
        raise RuntimeError("Falha ao consultar metadados do dataset 2014")

    mapping: dict[str, dict[str, str]] = {}
    for res in payload["result"].get("resources", []):
        name = str(res.get("name", "")).strip()
        rid = str(res.get("id", "")).strip()
        if not name or not rid:
            continue
        lower = name.lower()
        if "boletim de urna" not in lower or "hash" in lower:
            continue

        m = re.match(r"^([A-Z]{2})\s+-", name)
        if not m:
            continue
        uf = m.group(1)
        if uf not in UFS:
            continue

        if "primeiro turno" in lower:
            turno = "1t"
        elif "segundo turno" in lower:
            turno = "2t"
        else:
            continue
        mapping.setdefault(uf, {})[turno] = rid
    return mapping


def download_and_extract(resource_id: str, dest_csv: Path):
    url = f"https://dadosabertos.tse.jus.br/dataset/resultados-2014-boletim-de-urna/resource/{resource_id}/download"
    print(f"Baixando {url}")
    dest_csv.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        urllib.request.urlretrieve(url, tmp_path)
    try:
        with zipfile.ZipFile(tmp_path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise RuntimeError("ZIP sem CSV")
            with zf.open(csv_names[0]) as src, open(dest_csv, "wb") as dst:
                dst.write(src.read())
        print(f"  Extraido -> {dest_csv} ({dest_csv.stat().st_size} bytes)")
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    resources = load_resources()
    raw = Path("data/raw/2014")
    for uf in sorted(resources):
        turns = resources[uf]
        for turn in ("1t", "2t"):
            rid = turns.get(turn)
            if not rid:
                print(f"Pulando {uf} {turn} (sem resource ID)")
                continue
            name = f"bu_2014_{turn}_{uf}.csv"
            dest = raw / name
            if dest.exists() and is_valid_csv(dest):
                print(f"Ja valido: {dest}")
                continue
            try:
                if dest.exists():
                    dest.unlink()
                download_and_extract(rid, dest)
                if not is_valid_csv(dest):
                    print(f"  AVISO: {dest} ainda nao parece CSV valido")
            except Exception as e:
                print(f"  Falha em {uf} {turn}: {e}")
    print("Download 2014 concluido.")


if __name__ == "__main__":
    main()
