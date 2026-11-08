# Análise Descritiva – Eleições 2026

Estudo descritivo sobre o poder explicativo de regras simbólicas simples nos dados oficiais dos Boletins de Urna (BU) do TSE para as Eleições 2026.

**Escopo**: T1 (todas as votações) + ocorrências do T2 explicando T2 (todas as votações), no nível de seção.

Fonte dos dados: [Portal de Dados Abertos do TSE – Resultados 2026 – Boletim de Urna](https://dadosabertos.tse.jus.br/dataset/resultados-2026-boletim-de-urna)

## Estrutura

- `scripts/`: scripts de download, preparação e análise
- `data/raw/`: arquivos baixados (não versionados; ver `.gitignore`)
- `resultados/`: saídas processadas relevantes (métricas, regras, resíduos). **Sem arquivos crus.**

## Execução local em fila (self-hosted runner)

O workflow `.github/workflows/process.yml` está configurado para `runs-on: self-hosted`.

- Se o seu PC estiver **offline**, os jobs ficam em fila e nada é executado.
- Quando o runner estiver **online**, os jobs pendentes rodam automaticamente.

### Como instalar o runner no seu PC

1. No repositório, vá em **Settings → Actions → Runners → New self-hosted runner**.
2. Escolha o sistema operacional e siga os comandos que o GitHub mostra (download, config e run).
3. Deixe o processo do runner rodando (ou configure como serviço).

Depois disso, qualquer push que altere `scripts/` ou o workflow (ou um disparo manual) enfileira a execução no seu PC.

## Como usar manualmente

```bash
# Exemplo: baixar algumas UFs
python scripts/download_bu.py --ufs AC AL SP

# Ou todas
python scripts/download_bu.py --all

# Processar e gravar em resultados/
python scripts/process_and_summarize.py
```

Os arquivos crus ficam em `data/raw/`. Qualquer resultado processado deve ser gravado em `resultados/`.

Este repositório é para análise descritiva dos dados públicos. Não assume modificação de resultados.
