"""Task controller for git-triggered scope and quality validation."""

from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from jsonschema import ValidationError, validate


class TaskViolationError(RuntimeError):
    """Raised when a change violates task controller constraints."""


@dataclass(frozen=True)
class TaskSpec:
    """Machine-readable task policy used by controller checks."""

    task_id: str
    objective: str
    allowed_paths: List[str]
    blocked_paths: List[str]
    forbidden_regex: List[str]
    required_commands: List[str]
    forbidden_regex_exempt_paths: Optional[List[str]] = None
    precommit_commands: Optional[List[str]] = None
    ci_commands: Optional[List[str]] = None
    coverage_min: Optional[float] = None


class TaskController:
    """Enforce task constraints against git changes and required checks."""

    def __init__(self, repo_root: Optional[Path] = None):
        self.repo_root = repo_root or Path.cwd()

    @staticmethod
    def _normalize_mode(mode: str) -> str:
        normalized = mode.strip().lower()
        if normalized in {"pre-commit", "precommit"}:
            return "pre-commit"
        if normalized == "ci":
            return "ci"
        raise TaskViolationError(f"Unsupported controller mode: {mode}")

    def load_task_spec(self, task_file: Path) -> TaskSpec:
        """Load and validate a task spec file against schema."""
        resolved_task_file = task_file if task_file.is_absolute() else self.repo_root / task_file
        if not resolved_task_file.exists():
            raise TaskViolationError(f"Task file missing: {resolved_task_file}")

        with open(resolved_task_file, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        schema_path = self.repo_root / "config" / "schemas" / "task_spec.json"
        if not schema_path.exists():
            raise TaskViolationError(f"Task schema missing: {schema_path}")

        with open(schema_path, "r", encoding="utf-8") as handle:
            schema = json.load(handle)

        try:
            validate(instance=payload, schema=schema)
        except ValidationError as exc:
            raise TaskViolationError(f"Invalid task spec: {exc.message}") from exc

        return TaskSpec(
            task_id=payload["task_id"],
            objective=payload["objective"],
            allowed_paths=payload["allowed_paths"],
            blocked_paths=payload["blocked_paths"],
            forbidden_regex=payload["forbidden_regex"],
            forbidden_regex_exempt_paths=payload.get("forbidden_regex_exempt_paths"),
            required_commands=payload["required_commands"],
            precommit_commands=payload.get("precommit_commands"),
            ci_commands=payload.get("ci_commands"),
            coverage_min=payload.get("coverage_min"),
        )

    def get_commands_for_mode(self, spec: TaskSpec, mode: str) -> List[str]:
        """Resolve required commands for selected execution mode."""
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == "pre-commit" and spec.precommit_commands is not None:
            return spec.precommit_commands
        if normalized_mode == "ci" and spec.ci_commands is not None:
            return spec.ci_commands
        return spec.required_commands

    def _run_command(self, command: List[str]) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=self.repo_root,
            check=False,
        )

    def _run_shell(self, command: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            command,
            text=True,
            capture_output=True,
            cwd=self.repo_root,
            check=False,
            shell=True,
            executable="/bin/bash",
            env=os.environ.copy(),
        )

    def get_changed_files(self, staged: bool = False, diff_range: Optional[str] = None) -> List[str]:
        """Return changed files from git for staged or range-based checks."""
        if staged:
            command = ["git", "diff", "--cached", "--name-only"]
        elif diff_range:
            command = ["git", "diff", "--name-only", diff_range]
        else:
            command = ["git", "diff", "--name-only", "HEAD~1...HEAD"]

        result = self._run_command(command)
        if result.returncode != 0:
            raise TaskViolationError(f"Failed to read git diff: {result.stderr.strip()}")
        files = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        return files

    @staticmethod
    def _matches_prefix(path: str, prefixes: List[str]) -> bool:
        for prefix in prefixes:
            if prefix.endswith("/"):
                if path.startswith(prefix):
                    return True
            else:
                if path == prefix or path.startswith(prefix):
                    return True
        return False

    def validate_paths(self, changed_files: List[str], spec: TaskSpec) -> None:
        """Validate path scope and blocked paths."""
        for rel_path in changed_files:
            if spec.allowed_paths and not self._matches_prefix(rel_path, spec.allowed_paths):
                raise TaskViolationError(f"Out-of-scope file modified: {rel_path}")
            if spec.blocked_paths and self._matches_prefix(rel_path, spec.blocked_paths):
                raise TaskViolationError(f"Blocked path modified: {rel_path}")

    def _read_git_blob(self, rel_path: str) -> str:
        blob_result = self._run_command(["git", "show", f":{rel_path}"])
        if blob_result.returncode != 0:
            return ""
        return blob_result.stdout

    def validate_forbidden_patterns(self, changed_files: List[str], spec: TaskSpec, staged: bool = False) -> None:
        """Validate forbidden regex patterns in changed file content."""
        patterns = [re.compile(pattern, re.IGNORECASE) for pattern in spec.forbidden_regex]
        if not patterns:
            return

        for rel_path in changed_files:
            if spec.forbidden_regex_exempt_paths and self._matches_prefix(
                rel_path, spec.forbidden_regex_exempt_paths
            ):
                continue

            file_path = self.repo_root / rel_path
            if staged:
                content = self._read_git_blob(rel_path)
            elif file_path.exists() and file_path.is_file():
                content = file_path.read_text(encoding="utf-8", errors="ignore")
            else:
                content = ""

            for pattern in patterns:
                if pattern.search(content):
                    raise TaskViolationError(
                        f"Forbidden pattern '{pattern.pattern}' detected in {rel_path}"
                    )

    def run_required_commands(self, commands: List[str]) -> None:
        """Run required quality commands from task spec."""
        for command in commands:
            result = self._run_shell(command)
            if result.returncode != 0:
                message = result.stdout + "\n" + result.stderr
                raise TaskViolationError(f"Required command failed: {command}\n{message.strip()}")

    def run_coverage_gate(self, spec: TaskSpec) -> None:
        """Run enforced coverage gate when configured."""
        if spec.coverage_min is None:
            return

        coverage_command = (
            f"pytest tests/ -q --maxfail=1 --cov=src --cov-report=term-missing "
            f"--cov-fail-under={spec.coverage_min}"
        )
        result = self._run_shell(coverage_command)
        if result.returncode != 0:
            message = result.stdout + "\n" + result.stderr
            raise TaskViolationError(f"Coverage gate failed ({spec.coverage_min}%):\n{message.strip()}")

    def enforce(
        self,
        task_file: Path,
        staged: bool = False,
        diff_range: Optional[str] = None,
        mode: str = "ci",
    ) -> None:
        """Run all controller checks and raise on first violation."""
        normalized_mode = self._normalize_mode(mode)
        spec = self.load_task_spec(task_file)
        changed_files = self.get_changed_files(staged=staged, diff_range=diff_range)
        if not changed_files:
            return

        self.validate_paths(changed_files, spec)
        self.validate_forbidden_patterns(changed_files, spec, staged=staged)
        commands = self.get_commands_for_mode(spec, normalized_mode)
        self.run_required_commands(commands)
        if normalized_mode == "ci":
            self.run_coverage_gate(spec)
