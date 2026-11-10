#!/usr/bin/env python3
"""
Diagnóstico rápido dos arquivos baixados de 2018 e 2022.
Grava um resumo em resultados/diagnostico_csv.txt
"""

from pathlib import Path

def inspect(path: Path) -> str:
    if not path.exists():
        return f"{path.name}: NÃO EXISTE"
    size = path.stat().st_size
    try:
        with open(path, "rb") as f:
            head = f.read(300)
        text = head.decode("utf-8", errors="replace")
        preview = " | ".join(text.splitlines()[:3])[:200]
        has_html = "<html" in text.lower() or "<!doctype" in text.lower()
        has_semi = ";" in text
        return f"{path.name}: {size} bytes | HTML={has_html} | ;={has_semi} | preview: {preview!r}"
    except Exception as e:
        return f"{path.name}: ERRO ao ler ({e})"

def main():
    out = Path("resultados/diagnostico_csv.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["=== Diagnóstico CSVs 2018/2022 ===\n"]

    for ano in ("2018", "2022"):
        base = Path(f"data/raw/{ano}")
        lines.append(f"\n--- {ano} ---")
        if not base.exists():
            lines.append("  pasta não existe")
            continue
        files = sorted(base.glob("*.csv"))
        if not files:
            lines.append("  nenhum .csv encontrado")
            continue
        # Inspeciona os 8 primeiros para não ficar enorme
        for p in files[:8]:
            lines.append("  " + inspect(p))
        if len(files) > 8:
            lines.append(f"  ... e mais {len(files)-8} arquivos")

    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Diagnóstico gravado em {out}")
    print("\n".join(lines))

if __name__ == "__main__":
    main()
