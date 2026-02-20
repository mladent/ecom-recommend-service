"""Unit tests for task controller git validation."""

from pathlib import Path

import pytest

from src.task_controller import TaskController, TaskSpec, TaskViolationError


@pytest.mark.unit
class TestTaskController:
    """Task controller checks for scope, safety, and required gates."""

    def test_validate_paths_allows_in_scope_files(self, project_root: Path):
        controller = TaskController(repo_root=project_root)
        spec = TaskSpec(
            task_id="task-1",
            objective="scope test",
            allowed_paths=["src/", "tests/"],
            blocked_paths=[],
            forbidden_regex=[],
            required_commands=[],
            coverage_min=None,
        )

        controller.validate_paths(["src/task_controller.py", "tests/test_task_controller.py"], spec)

    def test_validate_paths_blocks_out_of_scope(self, project_root: Path):
        controller = TaskController(repo_root=project_root)
        spec = TaskSpec(
            task_id="task-1",
            objective="scope test",
            allowed_paths=["src/"],
            blocked_paths=[],
            forbidden_regex=[],
            required_commands=[],
            coverage_min=None,
        )

        with pytest.raises(TaskViolationError, match="Out-of-scope"):
            controller.validate_paths(["tests/test_task_controller.py"], spec)

    def test_validate_paths_blocks_blocked_path(self, project_root: Path):
        controller = TaskController(repo_root=project_root)
        spec = TaskSpec(
            task_id="task-1",
            objective="scope test",
            allowed_paths=["src/", "data/"],
            blocked_paths=["data/"],
            forbidden_regex=[],
            required_commands=[],
            coverage_min=None,
        )

        with pytest.raises(TaskViolationError, match="Blocked path"):
            controller.validate_paths(["data/data.csv"], spec)

    def test_validate_forbidden_patterns_detects_violation(self, project_root: Path, tmp_path: Path):
        file_path = tmp_path / "unsafe.py"
        file_path.write_text("os.system('rm -rf /tmp/example')", encoding="utf-8")

        controller = TaskController(repo_root=tmp_path)
        spec = TaskSpec(
            task_id="task-1",
            objective="pattern test",
            allowed_paths=["unsafe.py"],
            blocked_paths=[],
            forbidden_regex=[r"rm\s+-rf"],
            required_commands=[],
            coverage_min=None,
        )

        with pytest.raises(TaskViolationError, match="Forbidden pattern"):
            controller.validate_forbidden_patterns(["unsafe.py"], spec, staged=False)

    def test_run_required_commands_raises_on_failure(self, project_root: Path):
        controller = TaskController(repo_root=project_root)
        spec = TaskSpec(
            task_id="task-1",
            objective="checks test",
            allowed_paths=["src/"],
            blocked_paths=[],
            forbidden_regex=[],
            required_commands=["python -c 'import sys; sys.exit(2)'"],
            coverage_min=None,
        )

        with pytest.raises(TaskViolationError, match="Required command failed"):
            controller.run_required_commands(spec)

    def test_load_task_spec_from_repo_schema(self, project_root: Path):
        controller = TaskController(repo_root=project_root)
        spec = controller.load_task_spec(project_root / ".agent" / "current_task.json")

        assert spec.task_id
        assert isinstance(spec.allowed_paths, list)
        assert isinstance(spec.required_commands, list)

    def test_enforce_runs_all_validations(self, project_root: Path, monkeypatch):
        controller = TaskController(repo_root=project_root)
        calls = []

        def fake_get_changed_files(staged: bool = False, diff_range=None):
            calls.append("changed")
            return ["src/task_controller.py"]

        def fake_run_required_commands(spec: TaskSpec):
            calls.append("commands")

        def fake_run_coverage_gate(spec: TaskSpec):
            calls.append("coverage")

        monkeypatch.setattr(controller, "get_changed_files", fake_get_changed_files)
        monkeypatch.setattr(controller, "run_required_commands", fake_run_required_commands)
        monkeypatch.setattr(controller, "run_coverage_gate", fake_run_coverage_gate)

        controller.enforce(task_file=project_root / ".agent" / "current_task.json", staged=True)

        assert "changed" in calls
        assert "commands" in calls
        assert "coverage" in calls
