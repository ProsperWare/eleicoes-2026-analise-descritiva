# Análise Descritiva – Eleições 2026

Estudo descritivo sobre o poder explicativo de regras simbólicas simples nos dados oficiais dos Boletins de Urna (BU) do TSE para as Eleições 2026.

**Escopo**: T1 (todas as votações) + ocorrências do T2 explicando T2 (todas as votações), no nível de seção.

Fonte dos dados: [Portal de Dados Abertos do TSE – Resultados 2026 – Boletim de Urna](https://dadosabertos.tse.jus.br/dataset/resultados-2026-boletim-de-urna)

## Estrutura

- `scripts/`: scripts de download, preparação e análise
- `data/raw/`: arquivos baixados (não versionados; ver `.gitignore`)
- `resultados/`: saídas processadas relevantes (métricas, regras, resíduos). **Sem arquivos crus.**

## Como usar

```bash
# Exemplo: baixar algumas UFs
python scripts/download_bu.py --ufs AC AL SP

# Ou todas
python scripts/download_bu.py --all
```

Os arquivos crus ficam em `data/raw/`. Qualquer resultado processado (tabelas resumidas, regras extraídas, métricas de cobertura etc.) deve ser gravado em `resultados/`.

Este repositório é para análise descritiva dos dados públicos. Não assume modificação de resultados.
