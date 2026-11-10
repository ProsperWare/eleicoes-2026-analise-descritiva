#!/usr/bin/env python3
"""
Download Boletins de Urna (BU) oficiais das Eleições 2022 do TSE.
Fonte: https://dadosabertos.tse.jus.br/dataset/resultados-2022-boletim-de-urna

O endpoint devolve um ZIP contendo o CSV. Extrai o CSV e salva em data/raw/2022/.
"""

import urllib.request
import zipfile
import tempfile
from pathlib import Path

RESOURCES = {
    "AC": {"1t": "fdaba499-49eb-4bf7-be53-b34254efa38b", "2t": "fc894e35-4ab9-4ee7-a4e4-fc4658be8348"},
    "AL": {"1t": "68b1e117-66f0-46ac-bb84-b635709cfea4", "2t": "9974e5a8-9359-410d-86c0-611b811ceaf2"},
    "AM": {"1t": "388c4aed-577e-49ed-836b-dd9ea3da7ed3", "2t": "947cf29b-49ec-4310-9d68-7affbf944ab8"},
    "AP": {"1t": "8144dbb3-3d79-4623-afd5-590ecc74e0e2", "2t": "f38dcc7e-523c-497f-8ad9-a10e63004121"},
    "BA": {"1t": "c1881ece-aa58-4295-994d-c2d2c0ed486d", "2t": "5ddc12f6-afad-49bf-872a-cb2e0f3797fc"},
    "CE": {"1t": "ae3b27ba-46d8-453b-a1db-61f94e664ddc", "2t": "9b6629a6-1250-4a8e-a803-89d1adae77ba"},
    "DF": {"1t": "3a52e471-5054-4012-8867-d6c33fea92b8", "2t": "5649a238-1d7e-428b-8163-1d02cca9a837"},
    "ES": {"1t": "202229ee-352c-4942-a446-3234c125ca4b", "2t": "6eb7b097-ff3f-48f9-b11e-460cbdf34863"},
    "GO": {"1t": "e2086de4-031d-4e13-9d16-b053603f686c", "2t": "88d8d118-34b1-43d7-91ac-74ddb065e2c2"},
    "MA": {"1t": "be7af28f-f81f-49c1-b413-32bd84469dd0", "2t": "fb779e5f-a5f5-4738-9257-91feea600385"},
    "MG": {"1t": "d3c3d3e4-8b5b-47f0-97dc-56a6d5d1a8ce", "2t": "74f23cac-718f-440a-becf-305fd7f75da5"},
    "MS": {"1t": "bf8b2303-fe48-46b7-87cf-96f60a7ff158", "2t": "189bbf4e-1407-4547-a2a1-bb057b41bd88"},
    "MT": {"1t": "fd6558d9-0e6f-43a6-9c24-b1c35e77bbe7", "2t": "097dc255-be50-4931-8caa-9eca9f08af7c"},
    "PA": {"1t": "2f83840f-cf21-415a-9626-168d7d4deb7c", "2t": "d76d4bcf-790f-40e1-9c91-ff16f30016a3"},
    "PB": {"1t": "922ad74c-9b9b-4a5e-afff-d6279a1e8056", "2t": "ab8cc270-240c-4338-b073-fd816560408f"},
    "PE": {"1t": "f0aa6c0b-05b3-480d-bbb8-f403740830d8", "2t": "ae837fd2-1f5e-4739-bf10-80425eea8c27"},
    "PI": {"1t": "a01fabe5-1478-4e2c-b65c-835c165b9fc5", "2t": "a9973da0-3c69-4c1c-9f40-1767d9a7f6ad"},
    "PR": {"1t": "55bd5c37-369c-43b1-afa1-3d9cd0d51ffe", "2t": "e16f0285-8a5f-4625-adea-d949f16cc494"},
    "RJ": {"1t": "b1e24aec-ffce-4ffe-bcfb-37f8700ac7e5", "2t": "d5777bab-250c-45cc-bd7f-be539d7a320c"},
}

BASE = "https://dadosabertos.tse.jus.br/dataset/resultados-2022-boletim-de-urna/resource"

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

def download_and_extract(resource_id: str, dest_csv: Path):
    url = f"{BASE}/{resource_id}/download"
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
            # Pega o primeiro CSV (geralmente só tem um)
            with zf.open(csv_names[0]) as src, open(dest_csv, "wb") as dst:
                dst.write(src.read())
        print(f"  Extraído -> {dest_csv} ({dest_csv.stat().st_size} bytes)")
    finally:
        tmp_path.unlink(missing_ok=True)

def main():
    raw = Path("data/raw/2022")
    for uf, turns in RESOURCES.items():
        for turn, rid in turns.items():
            name = f"bu_2022_{turn}_{uf}.csv"
            dest = raw / name
            if dest.exists() and is_valid_csv(dest):
                print(f"Já válido: {dest}")
                continue
            try:
                if dest.exists():
                    dest.unlink()
                download_and_extract(rid, dest)
                if not is_valid_csv(dest):
                    print(f"  AVISO: {dest} ainda não parece CSV válido")
            except Exception as e:
                print(f"  Falha em {uf} {turn}: {e}")
    print("Download 2022 concluído.")

if __name__ == "__main__":
    main()
