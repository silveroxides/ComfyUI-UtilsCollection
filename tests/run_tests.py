"""Run exact tests or explicit groups; review changed-file coverage suggestions."""

# ruff: noqa: T201 - This module is an intentionally user-facing CLI.

from __future__ import annotations

import argparse
import fnmatch
import shutil
import subprocess
import sys
import tomllib
import uuid
from dataclasses import dataclass, field
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
COMFYUI_ROOT = REPOSITORY_ROOT.parent.parent
MANIFEST_PATH = Path(__file__).with_name("test_groups.toml")
TEMP_ROOT = REPOSITORY_ROOT / ".pytest-tmp"


@dataclass(frozen=True)
class TestGroup:
    name: str
    paths: tuple[str, ...]
    python_tests: tuple[str, ...]
    frontend_tests: tuple[str, ...]


@dataclass
class Selection:
    groups: set[str] = field(default_factory=set)
    python_tests: set[str] = field(default_factory=set)
    frontend_tests: set[str] = field(default_factory=set)
    reasons: dict[str, set[str]] = field(default_factory=dict)
    unmapped: set[str] = field(default_factory=set)


def _parse_groups(text: str) -> dict[str, TestGroup]:
    data = tomllib.loads(text)
    groups = {}
    for name, values in data.get("groups", {}).items():
        groups[name] = TestGroup(
            name=name,
            paths=tuple(values.get("paths", ())),
            python_tests=tuple(values.get("python_tests", ())),
            frontend_tests=tuple(values.get("frontend_tests", ())),
        )
    return groups


def load_groups(path: Path = MANIFEST_PATH) -> dict[str, TestGroup]:
    return _parse_groups(path.read_text(encoding="utf-8"))


def load_groups_from_revision(revision: str) -> dict[str, TestGroup]:
    manifest = MANIFEST_PATH.relative_to(REPOSITORY_ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"{revision}:{manifest}"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return _parse_groups(result.stdout) if result.returncode == 0 else {}


def git_lines(*args: str) -> set[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}


def changed_paths(base: str | None = None) -> set[str]:
    paths = git_lines("diff", "--name-only", "--relative", "HEAD")
    if base:
        paths.update(git_lines("diff", "--name-only", "--relative", f"{base}...HEAD"))
    return paths


def is_production_source(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return (
        ("/" not in normalized and normalized.endswith(".py"))
        or (normalized.startswith("web/") and normalized.endswith(".js"))
    )


def _add_group(selection: Selection, group: TestGroup, reason: str) -> None:
    selection.groups.add(group.name)
    selection.python_tests.update(group.python_tests)
    selection.frontend_tests.update(group.frontend_tests)
    selection.reasons.setdefault(group.name, set()).add(reason)


def _path_matches_group(path: str, group: TestGroup) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in group.paths)


def _existing_group(group: TestGroup) -> TestGroup:
    return TestGroup(
        name=group.name,
        paths=group.paths,
        python_tests=tuple(
            path for path in group.python_tests if (REPOSITORY_ROOT / path).is_file()
        ),
        frontend_tests=tuple(
            path for path in group.frontend_tests if (REPOSITORY_ROOT / path).is_file()
        ),
    )


def select_tests(
    paths: set[str],
    groups: dict[str, TestGroup],
    explicit_groups: tuple[str, ...] = (),
    historical_groups: tuple[dict[str, TestGroup], ...] = (),
) -> Selection:
    selection = Selection()
    for name in explicit_groups:
        if name not in groups:
            raise ValueError(f"Unknown test group: {name}")
        _add_group(selection, groups[name], "explicit selection")

    for raw_path in sorted(paths):
        path = raw_path.replace("\\", "/")
        if path.startswith("tests/test_") and path.endswith((".py", ".mjs")):
            if not (REPOSITORY_ROOT / path).is_file():
                continue
            if path.endswith(".py"):
                selection.python_tests.add(path)
            elif path.endswith(".mjs"):
                selection.frontend_tests.add(path)
            selection.reasons.setdefault("direct test", set()).add(path)
            continue

        matched = False
        for group in groups.values():
            if _path_matches_group(path, group):
                _add_group(selection, group, path)
                matched = True
        if not matched and not (REPOSITORY_ROOT / path).exists():
            for historical in historical_groups:
                for historical_group in historical.values():
                    if not _path_matches_group(path, historical_group):
                        continue
                    active_group = groups.get(historical_group.name)
                    _add_group(
                        selection,
                        active_group or _existing_group(historical_group),
                        f"{path} (deleted)",
                    )
                    matched = True
        if not matched and is_production_source(path) and (REPOSITORY_ROOT / path).exists():
            selection.unmapped.add(path)
    return selection


def tracked_final_tests() -> tuple[set[str], set[str]]:
    tracked = git_lines("ls-files", "tests/test_*.py", "tests/test_*.mjs")
    existing = {path for path in tracked if (REPOSITORY_ROOT / path).is_file()}
    python_tests = {path for path in existing if path.endswith(".py")}
    frontend_tests = {path for path in existing if path.endswith(".mjs")}
    return (
        python_tests,
        frontend_tests,
    )


def print_selection(selection: Selection, *, advisory: bool = False) -> None:
    print("Candidate coverage (suggestions only; no tests run):" if advisory else "Explicit execution selection:")
    for group in sorted(selection.groups):
        reasons = ", ".join(sorted(selection.reasons.get(group, ())))
        print(f"group {group}: {reasons}")
    for reason in sorted(selection.reasons.get("direct test", ())):
        print(f"direct test: {reason}")
    print("python tests:")
    for path in sorted(selection.python_tests):
        print(f"  {path}")
    print("frontend tests:")
    for path in sorted(selection.frontend_tests):
        print(f"  {path}")


def _resolved_tests(paths: set[str]) -> list[str]:
    resolved = []
    for target in sorted(paths):
        filename, separator, node_id = target.partition("::")
        path = (REPOSITORY_ROOT / filename).resolve()
        if not path.is_relative_to((REPOSITORY_ROOT / "tests").resolve()) or not path.is_file():
            raise ValueError(f"Test target must be an existing file under tests/: {target}")
        if path.suffix not in (".py", ".mjs") or (separator and (path.suffix != ".py" or not node_id)):
            raise ValueError(f"Unsupported test target: {target}")
        resolved.append(str(path) + separator + node_id)
    return resolved


def run_selection(selection: Selection) -> int:
    TEMP_ROOT.mkdir(exist_ok=True)
    basetemp = (TEMP_ROOT / f"run-{uuid.uuid4().hex}").resolve()
    if TEMP_ROOT.resolve() not in basetemp.parents:
        raise RuntimeError(f"Unsafe pytest temporary path: {basetemp}")
    try:
        if selection.python_tests:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "-q",
                    "--import-mode=importlib",
                    "--basetemp",
                    str(basetemp),
                    *_resolved_tests(selection.python_tests),
                ],
                cwd=COMFYUI_ROOT,
                check=False,
            )
            if result.returncode:
                return result.returncode
        if selection.frontend_tests:
            result = subprocess.run(
                ["node", "--test", *_resolved_tests(selection.frontend_tests)],
                cwd=REPOSITORY_ROOT,
                check=False,
            )
            if result.returncode:
                return result.returncode
        return 0
    finally:
        if basetemp.exists():
            shutil.rmtree(basetemp)
        if TEMP_ROOT.exists() and not any(TEMP_ROOT.iterdir()):
            TEMP_ROOT.rmdir()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--final", action="store_true", help="run every tracked test")
    mode.add_argument("--list-groups", action="store_true", help="list configured groups")
    mode.add_argument("--changed", action="store_true", help="suggest coverage for changed paths without running tests")
    mode.add_argument("--test", action="append", default=[], help="run only this test file or Python node ID; repeatable")
    mode.add_argument("--group", action="append", default=[], help="run an explicit group; repeatable")
    result.add_argument("--base", help="include committed changes since BASE")
    result.add_argument("--dry-run", action="store_true", help="explain without executing")
    return result


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    args = cli.parse_args(argv)
    if args.base and (args.test or args.group or args.final or args.list_groups):
        cli.error("--base is only supported for changed-file coverage suggestions")
    if not (args.test or args.group or args.final or args.list_groups or args.changed or args.base or args.dry_run):
        cli.print_help()
        return 0
    groups = load_groups() if args.list_groups or args.group or not (args.test or args.final) else {}
    if args.list_groups:
        for name in sorted(groups):
            print(name)
        return 0

    advisory = not (args.test or args.group or args.final)
    if args.test:
        try:
            targets = _resolved_tests(set(args.test))
        except ValueError as error:
            cli.error(str(error))
        selection = Selection(
            python_tests={target for target in targets if target.partition("::")[0].endswith(".py")},
            frontend_tests={target for target in targets if target.partition("::")[0].endswith(".mjs")},
        )
    elif args.final:
        python_tests, frontend_tests = tracked_final_tests()
        selection = Selection(
            groups={"final"},
            python_tests=python_tests,
            frontend_tests=frontend_tests,
            reasons={"final": {"every tracked test"}},
        )
    else:
        paths = changed_paths(args.base) if advisory else set()
        historical_groups = ()
        if paths:
            revisions = ["HEAD"]
            if args.base:
                revisions.append(args.base)
            historical_groups = tuple(load_groups_from_revision(revision) for revision in revisions)
        try:
            selection = select_tests(paths, groups, tuple(args.group), historical_groups)
        except ValueError as error:
            cli.error(str(error))

    if selection.unmapped:
        print("No mapped coverage recommendation for:")
        for path in sorted(selection.unmapped):
            print(f"  {path}")

    print_selection(selection, advisory=advisory)
    if not selection.python_tests and not selection.frontend_tests:
        print("No tests selected.")
        return 0
    return 0 if advisory or args.dry_run else run_selection(selection)


if __name__ == "__main__":
    raise SystemExit(main())
