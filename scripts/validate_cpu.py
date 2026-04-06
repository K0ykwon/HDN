from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, UTC
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "experiments" / "validation" / "cpu_basic"


def run_command(name: str, command: list[str]) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    return {
        "name": name,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    checks = [
        run_command("compileall", [sys.executable, "-m", "compileall", "src", "scripts", "tests"]),
        run_command(
            "pytest",
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/test_config_loading.py",
                "tests/test_latent_encoder.py",
                "tests/test_twr_forward.py",
                "tests/test_mamba_placeholder.py",
            ],
        ),
        run_command(
            "train_twr_backbone_debug",
            [
                sys.executable,
                "scripts/train.py",
                "--experiment",
                "configs/experiment/twr_backbone_lra_listops.yaml",
            ],
        ),
        run_command(
            "train_transformer_debug",
            [sys.executable, "scripts/train.py", "--experiment", "configs/experiment/transformer_debug.yaml"],
        ),
        run_command(
            "train_mamba_placeholder_debug",
            [sys.executable, "scripts/train.py", "--experiment", "configs/experiment/mamba_placeholder_debug.yaml"],
        ),
    ]

    report = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "python": sys.version,
        "checks": checks,
        "all_passed": all(check["returncode"] == 0 for check in checks),
    }

    with (RESULT_DIR / "validation_report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    with (RESULT_DIR / "validation_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"Validation timestamp: {report['timestamp_utc']}\n")
        handle.write(f"All passed: {report['all_passed']}\n\n")
        for check in checks:
            handle.write(f"[{check['name']}]\n")
            handle.write(f"command: {' '.join(check['command'])}\n")
            handle.write(f"returncode: {check['returncode']}\n")
            if check["stdout"]:
                handle.write("stdout:\n")
                handle.write(str(check["stdout"]))
                if not str(check["stdout"]).endswith("\n"):
                    handle.write("\n")
            if check["stderr"]:
                handle.write("stderr:\n")
                handle.write(str(check["stderr"]))
                if not str(check["stderr"]).endswith("\n"):
                    handle.write("\n")
            handle.write("\n")

    print(f"Saved validation report to {RESULT_DIR}")
    print(f"All passed: {report['all_passed']}")


if __name__ == "__main__":
    main()
