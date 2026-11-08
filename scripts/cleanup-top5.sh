#!/bin/bash
# cleanup-top5.sh
# Remove as 5 maiores pastas do /home/alexa identificadas na auditoria
# Uso:
#   ./cleanup-top5.sh          → dry-run (só mostra o que seria apagado)
#   ./cleanup-top5.sh --force  → apaga de verdade (pede confirmação)

set -euo pipefail

DRY_RUN=true
if [[ "${1:-}" == "--force" ]]; then
  DRY_RUN=false
fi

TARGETS=(
  "/home/alexa/Strata-data"
  "/home/alexa/eleicoes22"
  "/home/alexa/llama-models"
  "/home/alexa/vllm_openvino_venv"
  "/home/alexa/wsl-kernel"
)

echo "=========================================="
echo " Limpeza Top 5 pastas do /home/alexa"
echo "=========================================="
echo ""

for dir in "${TARGETS[@]}"; do
  if [[ -d "$dir" ]]; then
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    echo "  [OK] $dir  →  $size"
  else
    echo "  [MISSING] $dir  (já não existe)"
  fi
done

echo ""
echo "Espaço livre atual no / :"
df -h / | tail -1
echo ""

if $DRY_RUN; then
  echo "MODO DRY-RUN: nada será apagado."
  echo "Para apagar de verdade execute:"
  echo "  $0 --force"
  exit 0
fi

echo "ATENÇÃO: isso irá apagar permanentemente as pastas acima."
read -p "Digite 'APAGAR' para confirmar: " confirm
if [[ "$confirm" != "APAGAR" ]]; then
  echo "Cancelado."
  exit 1
fi

echo ""
echo "Iniciando exclusão..."
for dir in "${TARGETS[@]}"; do
  if [[ -d "$dir" ]]; then
    echo "Apagando $dir ..."
    rm -rf "$dir"
    echo "  → removido"
  fi
done

echo ""
echo "Limpeza concluída."
echo "Espaço livre agora no / :"
df -h / | tail -1
