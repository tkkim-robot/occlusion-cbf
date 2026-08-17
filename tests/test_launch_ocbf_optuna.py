"""Tests for the detached sequential Optuna launcher."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from tools import launch_ocbf_optuna as launcher


class LaunchOcbfOptunaTests(unittest.TestCase):
    def test_defaults_match_full_sequential_tuning_contract(self):
        args = launcher.parse_args([])

        self.assertEqual(args.trials, 100)
        self.assertIsNone(args.timeout)
        self.assertEqual(args.workers, 32)
        self.assertEqual(args.batch_size, 32)
        self.assertEqual(args.pruner_startup_trials, 12)
        self.assertEqual(args.pruner_warmup_cases, 64)
        self.assertEqual(args.pruner_interval_cases, 32)
        self.assertFalse(args.disable_pruning)
        self.assertFalse(args.wandb)
        launcher._validate_launcher_args(args)

    def test_source_manifest_hashes_sorted_relative_paths_and_contents(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "nested").mkdir()
            (root / "a.txt").write_text("alpha\n", encoding="utf-8")
            (root / "nested" / "b.txt").write_text(
                "beta\n", encoding="utf-8"
            )

            manifest = launcher._source_manifest(root)

            self.assertEqual(
                manifest["algorithm"],
                launcher.SOURCE_FINGERPRINT_ALGORITHM,
            )
            expected = hashlib.sha256()
            for relative in ("a.txt", "nested/b.txt"):
                digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
                expected.update(relative.encode("utf-8"))
                expected.update(b"\0")
                expected.update(bytes.fromhex(digest))
                expected.update(b"\n")
            self.assertEqual(manifest["fingerprint"], expected.hexdigest())

    def test_source_snapshot_copies_only_git_tracked_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "snapshot"
            git_result = SimpleNamespace(stdout=b"tracked.txt\0nested/link\0")
            source_file = launcher.REPO_ROOT / "tracked.txt"
            source_link = launcher.REPO_ROOT / "nested" / "link"

            def is_symlink(path: Path) -> bool:
                return path == source_link

            def is_file(path: Path) -> bool:
                return path == source_file

            with (
                mock.patch.object(
                    launcher.subprocess,
                    "run",
                    return_value=git_result,
                ),
                mock.patch.object(Path, "is_symlink", is_symlink),
                mock.patch.object(Path, "is_file", is_file),
                mock.patch.object(launcher.os, "readlink", return_value="target"),
                mock.patch.object(launcher.shutil, "copy2") as copy,
                mock.patch.object(Path, "symlink_to") as symlink,
            ):
                launcher._copy_tracked_source_snapshot(destination)

            copy.assert_called_once_with(
                source_file,
                destination / "tracked.txt",
            )
            symlink.assert_called_once_with("target")

    def test_study_command_omits_optional_timeout_and_wandb(self):
        command = launcher._study_command(
            python=Path("/venv/bin/python"),
            model="di",
            n_rand=50,
            workers=32,
            trials=100,
            timeout=None,
            batch_size=32,
            pruner_startup_trials=12,
            pruner_warmup_cases=64,
            pruner_interval_cases=32,
            disable_pruning=False,
            output_dir=Path("/results/di_n50"),
            study_name="di-study",
            wandb_project="unused",
            wandb_group="unused",
            wandb_enabled=False,
        )

        self.assertNotIn("taskset", command)
        self.assertNotIn("--timeout", command)
        self.assertNotIn("--wandb", command)
        self.assertNotIn("--disable-pruning", command)
        self.assertEqual(command[command.index("--trials") + 1], "100")
        self.assertEqual(command[command.index("--workers") + 1], "32")
        self.assertEqual(command[command.index("--batch-size") + 1], "32")
        self.assertEqual(
            command[command.index("--pruner-startup-trials") + 1], "12"
        )
        self.assertEqual(
            command[command.index("--pruner-warmup-cases") + 1], "64"
        )
        self.assertEqual(
            command[command.index("--pruner-interval-cases") + 1], "32"
        )
        self.assertEqual(command[command.index("--idx-start") + 1], "1")
        self.assertEqual(command[command.index("--idx-end") + 1], "100")

    def test_study_command_forwards_explicit_optional_settings(self):
        command = launcher._study_command(
            python=Path("/venv/bin/python"),
            model="uni",
            n_rand=30,
            workers=8,
            trials=5,
            timeout=120.0,
            batch_size=20,
            pruner_startup_trials=2,
            pruner_warmup_cases=40,
            pruner_interval_cases=20,
            disable_pruning=True,
            output_dir=Path("/results/uni_n30"),
            study_name="uni-study",
            wandb_project="project",
            wandb_group="group",
            wandb_enabled=True,
        )

        self.assertEqual(command[command.index("--timeout") + 1], "120.0")
        self.assertIn("--disable-pruning", command)
        self.assertIn("--wandb", command)
        self.assertEqual(
            command[command.index("--wandb-project") + 1], "project"
        )
        self.assertEqual(command[command.index("--wandb-group") + 1], "group")

    def test_validation_rejects_pruner_steps_that_batches_cannot_report(self):
        args = launcher.parse_args([])
        args.pruner_interval_cases = 10
        with self.assertRaisesRegex(ValueError, "multiple of --batch-size"):
            launcher._validate_launcher_args(args)

        args = launcher.parse_args([])
        args.pruner_warmup_cases = 50
        with self.assertRaisesRegex(ValueError, "align with a completed case batch"):
            launcher._validate_launcher_args(args)

        args = launcher.parse_args(["--timeout", "0"])
        with self.assertRaisesRegex(ValueError, "timeout.*positive"):
            launcher._validate_launcher_args(args)

    def test_validation_rejects_more_workers_than_available_cpus(self):
        args = launcher.parse_args([])
        with mock.patch.object(launcher, "_available_worker_count", return_value=16):
            with self.assertRaisesRegex(ValueError, "exceeds the 16 CPUs"):
                launcher._validate_launcher_args(args)

        args = launcher.parse_args(["--workers", "16"])
        with self.assertRaisesRegex(ValueError, "workers.*batch-size.*match"):
            launcher._validate_launcher_args(args)

    def test_launch_requires_clean_tracked_and_untracked_worktree(self):
        clean = SimpleNamespace(returncode=0, stdout="", stderr="")
        with mock.patch.object(launcher.subprocess, "run", return_value=clean) as run:
            launcher._require_clean_worktree()
        self.assertIn("--untracked-files=all", run.call_args.args[0])

        dirty = SimpleNamespace(
            returncode=0,
            stdout=" M tracked.py\n?? untracked.py\n",
            stderr="",
        )
        with mock.patch.object(launcher.subprocess, "run", return_value=dirty):
            with self.assertRaisesRegex(RuntimeError, "dirty worktree"):
                launcher._require_clean_worktree()

    @staticmethod
    def _write_supervisor_fixture(root: Path) -> Path:
        source_snapshot = root / "source_snapshot"
        logs_dir = root / "logs"
        source_snapshot.mkdir()
        logs_dir.mkdir()
        manifest_path = root / "launcher_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "queued",
                    "output_root": str(root),
                    "source_snapshot": str(source_snapshot),
                    "supervisor_log": str(logs_dir / "supervisor.log"),
                    "runs": [
                        {
                            "model": "di",
                            "status": "queued",
                            "command": ["python", "run-di"],
                            "log": str(logs_dir / "di.log"),
                        },
                        {
                            "model": "uni",
                            "status": "queued",
                            "command": ["python", "run-uni"],
                            "log": str(logs_dir / "uni.log"),
                        },
                    ],
                }
            )
            + "\n"
        )
        return manifest_path

    def test_supervisor_runs_studies_in_order_and_marks_complete(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = self._write_supervisor_fixture(Path(temp_dir))
            with mock.patch.object(
                launcher.subprocess,
                "run",
                side_effect=[
                    SimpleNamespace(returncode=0),
                    SimpleNamespace(returncode=0),
                ],
            ) as run_mock:
                result = launcher._supervise_manifest(manifest_path)

            self.assertEqual(result, 0)
            self.assertEqual(
                [call.args[0] for call in run_mock.call_args_list],
                [["python", "run-di"], ["python", "run-uni"]],
            )
            for call in run_mock.call_args_list:
                self.assertFalse(call.kwargs["check"])
                self.assertEqual(
                    call.kwargs["cwd"], Path(temp_dir) / "source_snapshot"
                )

            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "complete")
            self.assertEqual(manifest["supervisor"]["status"], "complete")
            self.assertEqual(
                [entry["status"] for entry in manifest["runs"]],
                ["complete", "complete"],
            )
            self.assertEqual(
                [entry["return_code"] for entry in manifest["runs"]],
                [0, 0],
            )
            self.assertIn("started_at", manifest)
            self.assertIn("completed_at", manifest)

    def test_supervisor_stops_after_failure_and_preserves_return_code(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = self._write_supervisor_fixture(Path(temp_dir))
            with mock.patch.object(
                launcher.subprocess,
                "run",
                return_value=SimpleNamespace(returncode=7),
            ) as run_mock:
                result = launcher._supervise_manifest(manifest_path)

            self.assertEqual(result, 7)
            run_mock.assert_called_once()
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["failed_run"], "di")
            self.assertEqual(manifest["runs"][0]["status"], "failed")
            self.assertEqual(manifest["runs"][0]["return_code"], 7)
            self.assertEqual(
                manifest["runs"][1]["status"],
                "not_started_after_failure",
            )

    def test_supervisor_records_exec_failure_without_starting_peer(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = self._write_supervisor_fixture(Path(temp_dir))
            with mock.patch.object(
                launcher.subprocess,
                "run",
                side_effect=OSError("cannot exec"),
            ) as run_mock:
                result = launcher._supervise_manifest(manifest_path)

            self.assertEqual(result, 1)
            run_mock.assert_called_once()
            manifest = json.loads(manifest_path.read_text())
            self.assertEqual(manifest["status"], "failed")
            self.assertIsNone(manifest["runs"][0]["return_code"])
            self.assertEqual(manifest["runs"][0]["launch_error"], "cannot exec")
            self.assertEqual(
                manifest["runs"][1]["status"],
                "not_started_after_failure",
            )

    def test_supervisor_rejects_log_path_outside_output_logs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = self._write_supervisor_fixture(root)
            manifest = json.loads(manifest_path.read_text())
            manifest["runs"][0]["log"] = str(root.parent / "outside.log")
            manifest_path.write_text(json.dumps(manifest) + "\n")

            with mock.patch.object(launcher.subprocess, "run") as run_mock:
                with self.assertRaisesRegex(ValueError, "inside output_root/logs"):
                    launcher._supervise_manifest(manifest_path)
            run_mock.assert_not_called()

    def test_supervisor_refuses_to_replay_a_started_manifest(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest_path = self._write_supervisor_fixture(Path(temp_dir))
            manifest = json.loads(manifest_path.read_text())
            manifest["status"] = "running"
            manifest_path.write_text(json.dumps(manifest) + "\n")

            with mock.patch.object(launcher.subprocess, "run") as run_mock:
                with self.assertRaisesRegex(RuntimeError, "not in queued state"):
                    launcher._supervise_manifest(manifest_path)
            run_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
