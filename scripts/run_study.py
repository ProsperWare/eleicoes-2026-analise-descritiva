#!/usr/bin/env python3
"""
Runner padronizado para estudos reproduziveis e handoff entre pessoas/agentes.

Uso:
  python scripts/run_study.py --study-id baseline-linear-residual
  python scripts/run_study.py --manifest estudos/baseline-linear-residual/study.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ESTUDOS_DIR = ROOT / "estudos"
RESULTS_DIR = ROOT / "resultados"


def _as_bool(value: str) -> bool:
    v = str(value).strip().lower()
    if v in {"1", "true", "yes", "y", "on"}:
        return True
    if v in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Booleano invalido: {value}")


def _as_list(value: str) -> list[str]:
    text = value.strip()
    if text.startswith("[") and text.endswith("]"):
        inner = text[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip('"\'') for item in inner.split(",") if item.strip()]
    return [item.strip() for item in text.split() if item.strip()]


def load_study_yaml(path: Path) -> dict:
    """Parser YAML minimo para manifestos simples chave/valor + listas."""
    data: dict[str, object] = {}
    current_list_key: str | None = None

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("- "):
            if current_list_key is None:
                raise ValueError(f"Lista sem chave em {path}: {raw}")
            data.setdefault(current_list_key, [])
            assert isinstance(data[current_list_key], list)
            data[current_list_key].append(line[2:].strip().strip('"\''))
            continue

        if ":" not in line:
            raise ValueError(f"Linha invalida em {path}: {raw}")

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        current_list_key = None

        if not value:
            data[key] = []
            current_list_key = key
            continue

        if value.startswith("[") and value.endswith("]"):
            data[key] = _as_list(value)
            continue

        low = value.lower()
        if low in {"true", "false", "yes", "no", "on", "off", "1", "0", "y", "n"}:
            data[key] = _as_bool(value)
            continue

        data[key] = value.strip('"\'')

    return data


def git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True)
        return out.strip()
    except Exception:
        return "unknown"


def to_cmd(manifest: dict) -> list[str]:
    anos = manifest.get("anos", ["2026"])
    if isinstance(anos, str):
        anos = _as_list(anos)
    if not anos:
        raise ValueError("Manifesto sem anos")

    cmd = [sys.executable, "scripts/orquestrar.py", "--anos", *[str(a) for a in anos]]
    for key, flag in (
        ("skip_download", "--skip-download"),
        ("skip_preprocess", "--skip-preprocess"),
        ("skip_residual", "--skip-residual"),
        ("skip_socio", "--skip-socio"),
        ("skip_benchmark", "--skip-benchmark"),
    ):
        if bool(manifest.get(key, False)):
            cmd.append(flag)
    ufs = manifest.get("ufs", [])
    if isinstance(ufs, str):
        ufs = _as_list(ufs)
    if ufs:
        cmd.extend(["--ufs", *[str(x) for x in ufs]])
    return cmd


def snapshot_outputs(study_id: str, run_tag: str, patterns: list[str]) -> list[str]:
    src_root = RESULTS_DIR
    dest = RESULTS_DIR / "estudos" / study_id / "snapshots" / run_tag
    dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []

    for pattern in patterns:
        for path in sorted(src_root.glob(pattern)):
            if not path.is_file():
                continue
            rel = path.relative_to(src_root)
            out = dest / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, out)
            copied.append(str(rel))
    return copied


@dataclass
class RunResult:
    exit_code: int
    start: str
    end: str
    duration_s: float
    command: list[str]
    copied_files: list[str]


def execute_study(study_id: str, manifest: dict) -> RunResult:
    run_tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    start_dt = datetime.now(timezone.utc)
    cmd = to_cmd(manifest)

    proc = subprocess.run(cmd, cwd=ROOT, check=False)

    end_dt = datetime.now(timezone.utc)
    patterns = manifest.get("output_patterns", [])
    if isinstance(patterns, str):
        patterns = _as_list(patterns)
    if not patterns:
        patterns = [
            "top2pct_urnas_*.csv",
            "comparacao_top2pct.txt",
            "intersecao_3anos*.txt",
            "intersecao_3anos.csv",
            "top_positivos_negativos.txt",
            "residual_*.csv",
            "residual_*.txt",
            "benchmark_modelos*.csv",
            "benchmark_modelos.md",
            "residual_ml_*.parquet",
        ]

    copied = snapshot_outputs(study_id, run_tag, [str(p) for p in patterns])
    return RunResult(
        exit_code=proc.returncode,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        duration_s=round((end_dt - start_dt).total_seconds(), 3),
        command=cmd,
        copied_files=copied,
    )


def write_run_json(study_id: str, manifest_path: Path, manifest: dict, result: RunResult):
    out_dir = RESULTS_DIR / "estudos" / study_id
    out_dir.mkdir(parents=True, exist_ok=True)
    run_path = out_dir / "run.json"

    payload = {
        "study_id": study_id,
        "manifest_path": str(manifest_path.relative_to(ROOT)),
        "git_commit": git_commit(),
        "status": "success" if result.exit_code == 0 else "failed",
        "start": result.start,
        "end": result.end,
        "duration_s": result.duration_s,
        "command": result.command,
        "anos": manifest.get("anos", []),
        "copied_files": result.copied_files,
        "exit_code": result.exit_code,
    }
    run_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def ensure_handoff(study_id: str):
    path = ESTUDOS_DIR / study_id / "HANDOFF.md"
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "# Handoff\n\n"
        "## Feito\n"
        "-\n\n"
        "## Pendente\n"
        "-\n\n"
        "## Riscos\n"
        "-\n\n"
        "## Proximo Comando\n"
        "- `python scripts/run_study.py --study-id " + study_id + "`\n",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Executa estudo com manifesto e gera run.json")
    parser.add_argument("--study-id", help="ID do estudo (usa estudos/<id>/study.yaml)")
    parser.add_argument("--manifest", type=Path, help="Caminho absoluto/relativo do manifesto")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.manifest:
        manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
        if not manifest_path.exists():
            raise SystemExit(f"Manifesto nao encontrado: {manifest_path}")
        study_id = args.study_id or manifest_path.parent.name
    else:
        if not args.study_id:
            raise SystemExit("Informe --study-id ou --manifest")
        study_id = args.study_id
        manifest_path = ESTUDOS_DIR / study_id / "study.yaml"
        if not manifest_path.exists():
            raise SystemExit(f"Manifesto nao encontrado: {manifest_path}")

    manifest = load_study_yaml(manifest_path)
    ensure_handoff(study_id)
    result = execute_study(study_id, manifest)
    write_run_json(study_id, manifest_path, manifest, result)

    print(f"Estudo: {study_id}")
    print(f"Manifesto: {manifest_path}")
    print(f"Status: {'OK' if result.exit_code == 0 else 'FALHOU'}")
    print(f"Duracao: {result.duration_s}s")
    print(f"Arquivos copiados: {len(result.copied_files)}")
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


if __name__ == "__main__":
    main()
