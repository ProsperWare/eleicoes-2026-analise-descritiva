#!/usr/bin/env python3
"""
Download Boletins de Urna (BU) oficiais das Eleições 2018 do TSE.
Fonte: https://dadosabertos.tse.jus.br/dataset/resultados-2018-boletim-de-urna

Os arquivos são CSVs por UF e turno. Coloque-os em data/raw/2018/.
"""

import urllib.request
from pathlib import Path

RESOURCES = {
    "AC": {"1t": "ceb9491d-3d99-4e69-ba78-43330503cab8", "2t": "e0cfecc9-39e8-4b52-af61-94bbd1ff15d8"},
    "AL": {"1t": "0cb5192e-d0b5-45dd-8684-2350c9627de9", "2t": "e3361229-78a4-40d7-8ecb-a093b8c7cf30"},
    "AM": {"1t": "e2199f02-4dcf-4e75-ad45-28e7a041f4cf", "2t": "65c35f91-2ea6-4876-a039-4bd15ee9c37d"},
    "AP": {"1t": "50ff44ee-fdba-4d5a-85b9-459b19000bb6", "2t": "19f88dd2-aa2e-457f-b077-0b4c07f31d15"},
    "BA": {"1t": "aa7a218e-2ca2-4d96-8dd2-918d82c07b8c", "2t": "a95b51ae-c744-4b00-93c2-0f65a50e6ced"},
    "CE": {"1t": "d6cb94a9-ad36-4beb-b5b2-f86ee7fff688", "2t": "aed47376-9962-41bd-a751-d32cc47ce631"},
    "DF": {"1t": "0f2fa59e-1886-45f7-a4bb-35543602af5f", "2t": "6ed70f8b-2a22-4ea5-a746-3e3424cf3590"},
    "ES": {"1t": "9ba3108a-b987-4086-9433-8b1358726c85", "2t": "ea3e31ee-78b7-4c65-9adb-bdda4ac39bb1"},
    "GO": {"1t": "04fd8e28-401e-4189-ad71-7257e833efb1", "2t": "cebabde7-6f9b-40ed-ae5a-b13dfe0331bf"},
    "MA": {"1t": "4e1a7622-04d3-4b11-aa4e-89a297e4cb7b", "2t": "cc744356-55a3-43a2-877e-2cf4cd8223b4"},
    "MG": {"1t": "896c6f54-98c5-4eab-907a-9d7828b3652e", "2t": "fd5e40b2-a26c-40c6-844c-7ef0793a6c07"},
    "MS": {"1t": "476802ac-1804-4df4-88aa-6ce1b81b57f5", "2t": "75155067-cd02-4dd3-bff3-1515b45f8228"},
    "MT": {"1t": "40d57624-91c1-4e94-8711-7e344b60df47", "2t": "b02a32e8-6706-42b4-ae14-5a2179b89c9e"},
    "PA": {"1t": "1615eb32-ebd0-4f4e-bfeb-d828aecae27b", "2t": "11b00e75-48de-4921-b648-dc19be18e0a2"},
    "PB": {"1t": "785c93c2-a6d1-43fb-a49e-f683bf462682", "2t": "e70ba944-ebf7-409d-a051-ba2e2968118b"},
    "PE": {"1t": "04c38962-fcf2-4d2d-b7ee-9f26def2a531", "2t": "757c5f5d-f939-412c-aa25-37b08a387f60"},
    "PI": {"1t": "eea7a508-947d-458f-b4ba-8eb9b6225b15", "2t": "812f3618-f279-4abf-901a-658139f1a84d"},
    "PR": {"1t": "14305a02-33fc-4efc-ab5b-8d867c944a13", "2t": "d9beb71a-288a-40b7-9237-b4d45f7beda1"},
    "RJ": {"1t": "35be51c9-a399-42fc-bf07-2b1d63ca4052", "2t": "63649b88-5725-4d95-ba03-838d229cf959"},
    "RN": {"1t": "f120a90b-32e8-453a-83d9-2363b908c63b", "2t": "af438440-5131-453e-8d34-00a8d1300cf9"},
    "RO": {"1t": "80fd8671-a362-46aa-a55d-1d9a48aac6a1", "2t": "f92cff03-c8e5-4549-b7f1-bb3aa17fc2ba"},
    "RR": {"1t": "a65fa011-4033-49f8-be83-952098126235", "2t": "0906ce0d-3feb-4d1e-84d5-cbcae2aa49ce"},
    "RS": {"1t": "452078b3-883e-43d0-80bc-bf1adbbd57ec", "2t": "59f3f172-3760-4152-ac62-260a1a433870"},
    "SC": {"1t": "985bfe56-56b0-4ae9-abe1-7f800925fa2d", "2t": "8efe92dc-4872-45c8-8065-2287bf93e4b9"},
    "SE": {"1t": "d049cf98-a8b5-460e-bff2-6a6462ecd81d", "2t": "8ee8d62d-1ef3-485f-86b1-50aea6061fe2"},
    "SP": {"1t": "c2974724-9b61-4f9c-9aca-0beb99c7bd8f", "2t": "903d309b-5486-41f5-869f-edad1034eeb1"},
    "TO": {"1t": "52898679-4f3c-42fa-8b49-8cf16f07ce60", "2t": None},  # 2t TO pode precisar de confirmação
}

BASE = "https://dadosabertos.tse.jus.br/dataset/resultados-2018-boletim-de-urna/resource"

def download_resource(resource_id: str, dest: Path):
    if not resource_id:
        return
    url = f"{BASE}/{resource_id}/download"
    print(f"Baixando {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)

def main():
    raw = Path("data/raw/2018")
    for uf, turns in RESOURCES.items():
        for turn, rid in turns.items():
            if rid is None:
                print(f"Pulando {uf} {turn} (sem resource ID)")
                continue
            name = f"bu_2018_{turn}_{uf}.csv"
            dest = raw / name
            if not dest.exists():
                try:
                    download_resource(rid, dest)
                except Exception as e:
                    print(f"  Falha em {uf} {turn}: {e}")
            else:
                print(f"Já existe: {dest}")
    print("Download 2018 concluído.")

if __name__ == "__main__":
    main()
