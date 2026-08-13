"""Model-config license/provenance contract check (DESIGN.md §2.4.4, issue #13).

DESIGN.md §2.4.4 requires every `configs/model/llm/*.yaml` to declare its
license and commercial-usability, and to pin a concrete revision rather than
a moving branch alias, so that:

  * the model's license is explicit in code review, not just in someone's
    memory of the model card, and
  * run metadata always resolves to a reproducible artifact instead of
    "whatever `main` happens to point to today".

This script is network-free: it only parses local YAML files. It is wired in
as both a pre-commit local hook and a standalone CLI
(`python scripts/check_model_configs.py [configs/model/llm]`).

Note: the corresponding pytest (tests/test_model_config_contract.py) and the
LLMConfig pydantic validator are owned by a different work unit; this script
intentionally does not depend on either.
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG_DIR = Path("configs/model/llm")
_DISALLOWED_REVISIONS = {"main", "master"}


class ConfigContractError(ValueError):
    """Raised when a model config violates the license/revision contract."""


def check_model_config(path: Path) -> list[str]:
    """Return a list of human-readable contract violations for one config file.

    An empty list means the file is compliant.
    """
    errors: list[str] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        return [f"{path}: could not parse YAML: {exc}"]

    if not isinstance(data, dict):
        return [f"{path}: expected a YAML mapping at the top level"]

    model_id = data.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        errors.append(f"{path}: `model_id` is missing or empty")

    revision = data.get("revision")
    if not isinstance(revision, str) or not revision.strip():
        errors.append(f"{path}: `revision` is missing or empty")
    elif revision.strip().lower() in _DISALLOWED_REVISIONS:
        errors.append(
            f"{path}: `revision: {revision}` is a moving branch alias, not a "
            "pinned commit/tag. Pin a concrete revision so run metadata stays "
            "reproducible even if the model is later replaced upstream "
            "(DESIGN.md §2.4.4)."
        )

    license_ = data.get("license")
    if not isinstance(license_, str) or not license_.strip():
        errors.append(
            f"{path}: `license` is missing or empty (DESIGN.md §2.4.4 requires "
            "every model config to declare its license explicitly)."
        )

    if "commercial_use_ok" not in data:
        errors.append(
            f"{path}: `commercial_use_ok` is missing (DESIGN.md §2.4.4 requires "
            "every model config to declare commercial-usability explicitly)."
        )

    return errors


def check_directory(config_dir: Path) -> dict[Path, list[str]]:
    """Check every *.yaml/*.yml file in config_dir. Returns path -> errors."""
    results: dict[Path, list[str]] = {}
    if not config_dir.is_dir():
        return results
    for path in sorted(config_dir.glob("*.yaml")) + sorted(config_dir.glob("*.yml")):
        errors = check_model_config(path)
        if errors:
            results[path] = errors
    return results


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    paths = [Path(a) for a in argv] if argv else [DEFAULT_CONFIG_DIR]

    all_errors: dict[Path, list[str]] = {}
    for p in paths:
        if p.is_dir():
            all_errors.update(check_directory(p))
        elif p.is_file():
            errors = check_model_config(p)
            if errors:
                all_errors[p] = errors
        else:
            print(f"warning: {p} does not exist, skipping", file=sys.stderr)

    if not all_errors:
        print("OK: all model configs satisfy the license/revision contract.")
        return 0

    for errors in all_errors.values():
        for err in errors:
            print(err)
    print(
        f"\n{sum(len(v) for v in all_errors.values())} violation(s) in "
        f"{len(all_errors)} file(s). See DESIGN.md §2.4.4."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
