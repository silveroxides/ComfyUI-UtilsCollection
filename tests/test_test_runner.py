import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


RUNNER_PATH = Path(__file__).with_name("run_tests.py")
SPEC = importlib.util.spec_from_file_location("utils_collection_test_runner", RUNNER_PATH)
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


def test_manifest_maps_every_tracked_production_source():
    groups = runner.load_groups()
    tracked = runner.git_lines("ls-files", "*.py", "web/*.js")
    production = {path for path in tracked if runner.is_production_source(path)}

    selection = runner.select_tests(production, groups)

    assert selection.unmapped == set()
    assert all(
        not path.startswith("scripts/")
        for group in groups.values()
        for path in group.paths
    )


def test_manifest_tests_are_tracked_and_present():
    groups = runner.load_groups()
    tracked = runner.git_lines("ls-files", "tests/test_*.py", "tests/test_*.mjs")
    configured = {
        path
        for group in groups.values()
        for path in (*group.python_tests, *group.frontend_tests)
    }

    assert configured <= tracked
    assert all((runner.REPOSITORY_ROOT / path).is_file() for path in configured)


def test_changed_paths_select_only_dependent_groups_and_direct_tests():
    groups = runner.load_groups()
    selection = runner.select_tests(
        {
            "staged_compositor_helpers.py",
            "encoder_helpers.py",
            "tests/test_scheduler_migration.py",
            "README.md",
        },
        groups,
    )

    assert selection.groups == {"composite", "encoder", "minimax_h3_cache"}
    assert "tests/test_composite_nodes.py" in selection.python_tests
    assert "tests/test_advanced_visual_consensus.py" in selection.python_tests
    assert "tests/test_minimax_h3_cache.py" in selection.python_tests
    assert "tests/test_scheduler_migration.py" in selection.python_tests
    assert selection.frontend_tests == set()


@pytest.mark.parametrize(
    ("path", "group_names", "python_tests"),
    (
        (
            "encoder_helpers.py",
            {"encoder", "minimax_h3_cache"},
            {
                "tests/test_advanced_visual_consensus.py",
                "tests/test_encoder_correctness.py",
                "tests/test_visual_fusion.py",
                "tests/test_minimax_h3_cache.py",
            },
        ),
        (
            "vlm_presets.py",
            {"vlm"},
            {
                "tests/test_vlm_presets.py",
                "tests/test_vlm_preset_authorities.py",
            },
        ),
        (
            "minimax_h3_vlm_presets.py",
            {"minimax_h3_vlm"},
            {
                "tests/test_minimax_h3_vlm_presets.py",
                "tests/test_vlm_preset_authorities.py",
            },
        ),
        (
            "presets_collection.py",
            {"presets"},
            {
                "tests/test_photography_presets.py",
                "tests/test_preset_parentheses.py",
                "tests/test_video_prompt_conversion.py",
            },
        ),
    ),
)
def test_subsystem_changes_select_only_relevant_tests(path, group_names, python_tests):
    selection = runner.select_tests({path}, runner.load_groups())

    assert selection.groups == group_names
    assert selection.python_tests == python_tests
    assert selection.frontend_tests == set()


def test_manifest_change_uses_test_selector_group():
    selection = runner.select_tests({"tests/test_groups.toml"}, runner.load_groups())

    assert selection.groups == {"test_selector"}
    assert selection.python_tests == {"tests/test_test_runner.py"}


def test_revision_manifest_is_loaded_from_requested_base(monkeypatch):
    calls = []
    manifest = """
[groups.encoder]
paths = ["deleted.py"]
python_tests = ["tests/test_encoder.py"]
frontend_tests = []
"""
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or subprocess.CompletedProcess(command, 0, stdout=manifest, stderr=""),
    )

    groups = runner.load_groups_from_revision("release-base")

    assert calls[0][0] == ["git", "show", "release-base:tests/test_groups.toml"]
    assert groups["encoder"].paths == ("deleted.py",)


def test_frontend_source_selects_frontend_and_parity_coverage():
    selection = runner.select_tests(
        {"web/layered_background_editor.js"}, runner.load_groups()
    )

    assert selection.groups == {"staged_frontend"}
    assert "tests/test_composite_nodes.py" in selection.python_tests
    assert "tests/test_layered_placement.mjs" in selection.frontend_tests
    assert "tests/test_staged_editor_layout.mjs" in selection.frontend_tests


def test_unknown_production_source_is_an_advisory_gap(monkeypatch, tmp_path, capsys):
    (tmp_path / "new_domain.py").touch()
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    selection = runner.select_tests({"new_domain.py"}, runner.load_groups())

    assert selection.unmapped == {"new_domain.py"}
    monkeypatch.setattr(runner, "changed_paths", lambda base: {"new_domain.py"})
    monkeypatch.setattr(runner, "load_groups_from_revision", lambda revision: {})
    monkeypatch.setattr(runner, "run_selection", lambda selection: pytest.fail("Advisory mode ran tests"))
    assert runner.main(["--changed"]) == 0
    output = capsys.readouterr().out
    assert "new_domain.py" in output and "no tests run" in output
    assert "--final" not in output


def test_deleted_source_uses_historical_group_without_deleted_test(monkeypatch, tmp_path):
    current_test = tmp_path / "tests" / "test_current.py"
    current_test.parent.mkdir()
    current_test.touch()
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    current = {
        "encoder": runner.TestGroup(
            "encoder", ("encoder_nodes.py",), ("tests/test_current.py",), ()
        )
    }
    historical = {
        "encoder": runner.TestGroup(
            "encoder",
            ("encoder_nodes.py", "qwen_vlm_nodes.py"),
            ("tests/test_current.py", "tests/test_deleted.py"),
            (),
        )
    }

    selection = runner.select_tests(
        {"qwen_vlm_nodes.py"}, current, historical_groups=(historical,)
    )

    assert selection.groups == {"encoder"}
    assert selection.python_tests == {"tests/test_current.py"}
    assert selection.unmapped == set()


def test_deleted_direct_test_is_not_selected(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)

    selection = runner.select_tests({"tests/test_deleted.py"}, {})

    assert selection.python_tests == set()
    assert selection.reasons == {}


def test_explicit_group_and_unknown_group_behavior():
    groups = runner.load_groups()
    selection = runner.select_tests(set(), groups, ("tiling",))

    assert selection.groups == {"tiling"}
    assert selection.python_tests == {
        "tests/test_high_resolution_tiling.py",
        "tests/test_high_resolution_tiling_guide.py",
    }
    with pytest.raises(ValueError, match="Unknown test group"):
        runner.select_tests(set(), groups, ("missing",))


def test_final_test_discovery_excludes_deleted_tests(monkeypatch, tmp_path):
    python_test = tmp_path / "tests" / "test_one.py"
    frontend_test = tmp_path / "tests" / "test_two.mjs"
    python_test.parent.mkdir()
    python_test.touch()
    frontend_test.touch()
    monkeypatch.setattr(runner, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(
        runner,
        "git_lines",
        lambda *args: {
            "tests/test_one.py",
            "tests/test_two.mjs",
            "tests/test_deleted.py",
        },
    )

    assert runner.tracked_final_tests() == (
        {"tests/test_one.py"},
        {"tests/test_two.mjs"},
    )


def test_run_selection_uses_configured_interpreter_and_cleans_temp(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "TEMP_ROOT", tmp_path / "pytest-temp")
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda command, **kwargs: calls.append((command, kwargs))
        or subprocess.CompletedProcess(command, 0),
    )
    selection = runner.Selection(
        python_tests={"tests/test_bounding_box.py"},
        frontend_tests={"tests/test_layered_placement.mjs"},
    )

    assert runner.run_selection(selection) == 0
    assert calls[0][0][:4] == [runner.sys.executable, "-m", "pytest", "-q"]
    assert calls[0][1]["cwd"] == runner.COMFYUI_ROOT
    assert calls[1][0][:2] == ["node", "--test"]
    assert not runner.TEMP_ROOT.exists()


def test_exact_targets_preserve_node_ids_without_expansion_and_propagate_exit(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(runner, "TEMP_ROOT", tmp_path / "pytest-temp")
    monkeypatch.setattr(runner, "changed_paths", lambda base: pytest.fail("Exact selection inspected changes"))
    monkeypatch.setattr(runner, "load_groups", lambda: pytest.fail("Exact selection expanded groups"))
    monkeypatch.setattr(runner.subprocess, "run", lambda command, **kwargs:
                        calls.append(command) or subprocess.CompletedProcess(command, 5))
    targets = ["tests/test_test_runner.py::TestClass::test_behavior[case::value.mjs]",
               "tests/test_test_runner.py::test_other"]
    assert runner.main(["--test", targets[0], "--test", targets[1]]) == 5
    assert len(calls) == 1
    expected = [str(Path(__file__).resolve()) + "::" + value.partition("::")[2] for value in targets]
    assert calls[0][-2:] == expected
    assert not runner.TEMP_ROOT.exists()


def test_help_and_advisory_modes_never_run_tests(monkeypatch, capsys):
    monkeypatch.setattr(runner, "run_selection", lambda selection: pytest.fail("Unexpected execution"))
    monkeypatch.setattr(runner, "changed_paths", lambda base: pytest.fail("No-argument invocation inspected changes"))
    assert runner.main([]) == 0
    assert "--test" in capsys.readouterr().out
    monkeypatch.setattr(runner, "changed_paths", lambda base: {"encoder_helpers.py"})
    monkeypatch.setattr(runner, "load_groups_from_revision", lambda revision: {})
    for args in (["--changed"], ["--dry-run"], ["--base", "HEAD"]):
        assert runner.main(args) == 0
        assert "Candidate coverage (suggestions only; no tests run)" in capsys.readouterr().out


@pytest.mark.parametrize("args", [
    ["--test", "tests/missing_test.py"], ["--test", "tests"],
    ["--test", "tests/test_groups.toml"], ["--test", "tests/test_load_image_alpha.mjs::case"],
    ["--test", "tests/test_test_runner.py", "--changed"],
    ["--group", "encoder", "--base", "HEAD"], ["--group", "unknown-group"],
])
def test_invalid_explicit_selection_never_runs_a_fallback(args, monkeypatch):
    monkeypatch.setattr(runner, "run_selection", lambda selection: pytest.fail("Unexpected execution"))
    with pytest.raises(SystemExit) as error:
        runner.main(args)
    assert error.value.code == 2


def test_explicit_groups_final_and_dry_run_keep_requested_scope(monkeypatch):
    calls = []
    monkeypatch.setattr(runner, "changed_paths", lambda base: pytest.fail("Explicit selection inspected changes"))
    monkeypatch.setattr(runner, "run_selection", lambda selection: calls.append(selection) or 0)
    monkeypatch.setattr(runner, "tracked_final_tests", lambda: ({"tests/test_test_runner.py"}, set()))
    assert runner.main(["--group", "minimax_h3_cache", "--group", "minimax_h3_guide"]) == 0
    assert calls[-1].groups == {"minimax_h3_cache", "minimax_h3_guide"}
    assert runner.main(["--final"]) == 0
    assert calls[-1].groups == {"final"}
    for args in (["--final", "--dry-run"], ["--group", "encoder", "--dry-run"],
                 ["--test", "tests/test_test_runner.py", "--dry-run"]):
        assert runner.main(args) == 0
    assert len(calls) == 2
