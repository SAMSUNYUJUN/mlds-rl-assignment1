"""Audit saved assignment evidence without training, publishing, or contacting GitHub.

Writes data/delivery_check.json and ACCEPTANCE.md. The default exit status gates
local scientific deliverables. --require-remote additionally requires the explicit
remote_code_delivered flag in data/repository.json. Run this checker before the
final package rebuild so the archive includes the current acceptance evidence.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from urllib.parse import urlparse
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RUNS = ("cartpole_basic", "cartpole_baseline", "pong_basic", "pong_baseline")
FIGURES = ["shuttle_enumeration.png", "shuttle_value_iteration.png", "shuttle_policy_iteration.png"] + [
    f"{name}_{kind}.png" for name in RUNS for kind in ("training", "evaluation")
]
PLACEHOLDERS = [
    r"\bTODO\b", r"\bTBD\b", r"\bYOUR[_ -](?:GITHUB|USERNAME|REPO(?:SITORY)?|NAME)\b",
    r"<(?:github[_ -]?(?:username|url)|username|repository[_ -]?(?:name|url))>",
    r"\[INSERT[^\]]*\]", r"\bINSERT_(?:GITHUB|USERNAME|REPO)\b",
]


def read_json(path: Path):
    return json.loads(path.read_text())


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def ensure(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def close(actual, expected, label: str, atol=1e-9, rtol=1e-10) -> None:
    ensure(math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol),
           f"{label}: saved {actual!r}, independently recomputed {expected!r}")


def truth(value) -> bool:
    if value is True or str(value).strip().lower() in {"true", "1"}:
        return True
    if value is False or str(value).strip().lower() in {"false", "0"}:
        return False
    raise ValueError(f"Invalid Boolean value {value!r}")


def read_episode_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    ensure(bool(records), f"{path}: no episode rows")
    rows = []
    total = 0
    last_elapsed = -1.0
    for i, record in enumerate(records):
        row = {key: int(record[key]) for key in ("episode", "seed", "length", "env_steps_total")}
        row.update({key: float(record[key]) for key in ("reward", "discounted_return", "elapsed_seconds")})
        row.update({key: truth(record[key]) for key in ("terminated", "truncated")})
        row["checkpoint"] = record["checkpoint"]
        ensure(row["episode"] == i, f"{path}: episode IDs must occur once in order 0..n-1; row {i}")
        ensure(row["length"] > 0, f"{path}: nonpositive episode length at row {i}")
        ensure(row["terminated"] or row["truncated"], f"{path}: incomplete episode at row {i}")
        ensure(all(math.isfinite(row[key]) for key in ("reward", "discounted_return", "elapsed_seconds")),
               f"{path}: nonfinite number at row {i}")
        total += row["length"]
        ensure(row["env_steps_total"] == total, f"{path}: inconsistent cumulative environment steps at row {i}")
        ensure(row["elapsed_seconds"] >= max(0.0, last_elapsed), f"{path}: decreasing or negative elapsed time at row {i}")
        last_elapsed = row["elapsed_seconds"]
        rows.append(row)
    ensure(len({row["seed"] for row in rows}) == len(rows), f"{path}: duplicate episode seed")
    return rows


def episode_statistics(rows: list[dict]) -> dict:
    """Use Python's statistics module independently of training/plot NumPy code."""
    rewards = [row["reward"] for row in rows]
    lengths = [row["length"] for row in rows]
    return {
        "n": len(rows), "mean": statistics.mean(rewards),
        "std_sample": statistics.stdev(rewards) if len(rewards) > 1 else None,
        "std_population": statistics.pstdev(rewards), "min": min(rewards), "max": max(rewards),
        "mean_length": statistics.mean(lengths), "env_steps": sum(lengths),
        "terminated": sum(row["terminated"] for row in rows),
        "truncated": sum(row["truncated"] for row in rows),
        "elapsed_seconds": rows[-1]["elapsed_seconds"],
    }


class Audit:
    def __init__(self, root: Path, require_remote: bool):
        self.root, self.require_remote = root, require_remote
        self.checks = []
        self.runs = {}
        self.repository = None
        self.report_pages = None

    def add(self, check_id, requirement, evidence, function, required_local=True, scope="local"):
        try:
            details = function()
            passed, error = True, None
        except Exception as exc:
            details, passed, error = None, False, f"{type(exc).__name__}: {exc}"
        self.checks.append(dict(id=check_id, requirement=requirement, evidence=evidence,
                                passed=passed, required_local=required_local, scope=scope,
                                details=details, error=error))

    def source_pdf(self):
        from pypdf import PdfReader
        reader = PdfReader(str(self.root / "mlds_assignment1.pdf"))
        ensure(len(reader.pages) == 4, "The assignment source must have the four supplied pages")
        extracted = (self.root / "docs/assignment_extracted.txt").read_text()
        markers = list(re.finditer(r"(?m)^PAGE (\d+)\s*$", extracted))
        ensure([int(marker[1]) for marker in markers] == [1, 2, 3, 4], "Extracted text must label all four pages in order")
        for i, marker in enumerate(markers):
            end = markers[i + 1].start() if i + 1 < len(markers) else len(extracted)
            saved = re.sub(r"\s+", "", extracted[marker.end():end])
            actual = re.sub(r"\s+", "", reader.pages[i].extract_text() or "")
            ensure(bool(actual) and actual == saved, f"Saved extraction differs from source page {i + 1}")
        return {"pages": 4, "source_sha256": digest(self.root / "mlds_assignment1.pdf"),
                "all_saved_page_text_matches_source": True}

    def document_files(self):
        paths = ["CODEX_TASK.md", "USER_CONTEXT.md", "REQUIREMENTS.md", "PROGRESS.md", "README.md",
                 "requirements.txt", "requirements-lock.txt", "docs/LEARNING_ZH.md", "docs/report.md",
                 "docs/shuttle_method.md", "docs/rl_validation.md", "docs/experiment_budget.md",
                 "src/shuttle.py", "src/rl.py", "src/validate_rl.py", "src/plot_results.py",
                 "src/build_report.py", "scripts/run_experiments.py", "scripts/package.py", "scripts/check_delivery.py"]
        missing = [path for path in paths if not (self.root / path).is_file() or (self.root / path).stat().st_size == 0]
        ensure(not missing, "Missing/empty deliverables: " + ", ".join(missing))
        readme = (self.root / "README.md").read_text()
        for token in ["shuttle.py", "rl.py", "evaluate", "run_experiments.py", "validate_rl.py",
                      "plot_results.py", "build_report.py", "requirements-lock.txt"]:
            ensure(token in readme, f"README omits run/setup entry {token}")
        chinese = (self.root / "docs/LEARNING_ZH.md").read_text()
        ensure(re.search(r"[\u4e00-\u9fff]", chinese) is not None, "Chinese learning guide contains no Chinese text")
        lock = (self.root / "requirements-lock.txt").read_text()
        for package in ["torch", "gymnasium", "ale-py", "numpy", "matplotlib", "reportlab", "pypdf"]:
            ensure(re.search(rf"(?mi)^{re.escape(package)}==\S+", lock) is not None, f"Lockfile has no exact {package} version")
        return {"nonempty_required_files": paths, "readme_command_entries_checked": True,
                "chinese_guide_present": True, "dependency_versions_pinned": True}

    def plan(self):
        plan = read_json(self.root / "data/experiment_plan.json")
        run_list = plan["runs"]
        ensure(len(run_list) == 4 and {run["name"] for run in run_list} == set(RUNS), "Formal plan must contain the four canonical runs exactly once")
        for run in run_list:
            ensure(isinstance(run["episodes"], int) and run["episodes"] > 0, f"Invalid formal budget for {run['name']}")
        return {"runs": run_list, "path": "data/experiment_plan.json"}

    def runtime(self):
        hardware = read_json(self.root / "data/hardware.json")
        env = read_json(self.root / "data/environment.json")
        ensure("M4 Pro" in hardware["chip"] and int(hardware["memory_gb"]) == 24, "Saved hardware differs from the user's Mac")
        ensure(env["machine"] == "arm64" and "macOS" in env["platform"], "Actual saved Python runtime is not native arm64 macOS")
        ensure("CartPole-v1" in env["cartpole_spec"] and "ALE/Pong-v5" in env["pong_spec"], "Required registered environment versions missing")
        ensure(env["pong_action_meanings"][2:4] == ["RIGHT", "LEFT"], "Pong action IDs 2/3 do not match required meanings")
        ensure(env.get("pong_ram_input") is False, "Pong image-input verification missing")
        return {"hardware": hardware, "registered_action_pair": env["pong_action_meanings"][2:4],
                "platform": env["platform"], "pong_frameskip": env["pong_frameskip"],
                "sticky_actions": env["sticky_actions"]}

    def benchmark(self):
        decision = read_json(self.root / "data/benchmark/decision.json")
        summaries = decision["summaries"]
        devices = {summary["device"] for summary in summaries}
        ensure("cpu" in devices, "No actual CPU benchmark")
        if decision["mps_available"]:
            ensure("mps" in devices, "MPS was available but no actual MPS benchmark was saved")
        for summary in summaries:
            ensure(float(summary["seconds"]) >= 300, f"{summary['device']}: benchmark is shorter than the recorded 300-second budget")
            ensure(int(summary["environment_steps"]) > 0 and int(summary["process_peak_rss_bytes"]) > 0, "Benchmark steps/memory missing")
            close(summary["steps_per_second"], summary["environment_steps"] / summary["seconds"], "Benchmark throughput")
            ensure(float(summary["update_seconds_mean"]) >= 0, "Benchmark update timing missing")
        selected = max(summaries, key=lambda summary: summary["steps_per_second"])["device"]
        ensure(decision["selected"] == selected, "Device choice differs from recorded throughput criterion")
        return {"selected": selected, "devices": sorted(devices), "actual_summaries": summaries}

    def run_config(self, name):
        folder = self.root / "runs" / name
        cfg = read_json(folder / "config.json")
        plan = read_json(self.root / "data/experiment_plan.json")
        item = next(run for run in plan["runs"] if run["name"] == name)
        for key in ["game", "baseline", "seed", "episodes", "lr", "batch_size"]:
            ensure(cfg[key] == item[key], f"{name}: {key} differs between formal plan and saved configuration")
        ensure(cfg["game"] == name.split("_")[0] and cfg["baseline"] == name.endswith("baseline"), f"{name}: mislabeled algorithm/environment")
        close(cfg["gamma"], 0.95 if cfg["game"] == "cartpole" else 0.99, f"{name} discount factor")
        ensure(cfg["device"] == "cpu" if cfg["game"] == "cartpole" else cfg["device"] in {"cpu", "mps"}, f"{name}: invalid local device")
        self.runs.setdefault(name, {})["config"] = cfg
        return {"configuration_matches_plan": True, "episodes": cfg["episodes"], "seed": cfg["seed"],
                "device": cfg["device"], "lr": cfg["lr"], "gamma": cfg["gamma"]}

    def training(self, name):
        folder = self.root / "runs" / name
        cfg = read_json(folder / "config.json")
        rows = read_episode_csv(folder / "train.csv")
        ensure(len(rows) == int(cfg["episodes"]), f"{name}: actual training episode count differs from formal budget")
        planned = next(run for run in read_json(self.root / "data/experiment_plan.json")["runs"] if run["name"] == name)
        ensure(len(rows) == int(planned["episodes"]), f"{name}: train.csv differs from preregistered formal budget")
        ensure(all(row["seed"] == int(cfg["train_seed_base"]) + row["episode"] for row in rows), f"{name}: noncontiguous training seed sequence")
        stats = episode_statistics(rows)
        stats["last100_mean_reward"] = statistics.mean(row["reward"] for row in rows[-100:])
        self.runs.setdefault(name, {}).update(training=stats, train_seeds=[row["seed"] for row in rows])
        return stats

    def checkpoint(self, name):
        folder = self.root / "runs" / name
        cfg = read_json(folder / "config.json")
        frozen = read_json(folder / "FROZEN.json")
        actual = digest(folder / "final.pt")
        ensure(frozen["file"] == "final.pt" and frozen["sha256"] == actual, f"{name}: frozen checkpoint SHA mismatch")
        ensure(int(frozen["episode"]) == int(cfg["episodes"]), f"{name}: checkpoint freeze count differs from formal budget")
        ensure(bool(frozen.get("rule")), f"{name}: no checkpoint-selection rule")
        import torch
        saved = torch.load(folder / "final.pt", map_location="cpu", weights_only=False)
        ensure(saved["config"] == cfg, f"{name}: final checkpoint's embedded config differs from config.json")
        ensure(int(saved["episode"]) == int(cfg["episodes"]), f"{name}: saved checkpoint episode differs from budget")
        rows = read_episode_csv(folder / "train.csv")
        ensure(int(saved["steps"]) == sum(row["length"] for row in rows), f"{name}: checkpoint step count differs from raw training")
        self.runs.setdefault(name, {})["checkpoint_sha256"] = actual
        return {"sha256": actual, "episode": frozen["episode"], "embedded_config_matches": True,
                "selection_rule": frozen["rule"], "freeze_unix_seconds": frozen.get("frozen_at")}

    def evaluation(self, name):
        folder = self.root / "runs" / name
        cfg = read_json(folder / "config.json")
        rows = read_episode_csv(folder / "evaluation.csv")
        ensure(len(rows) == 500, f"{name}: require exactly 500 complete environment episodes; found {len(rows)}")
        ensure(all(row["seed"] == int(cfg["eval_seed_base"]) + row["episode"] for row in rows), f"{name}: evaluation seeds differ from configured consecutive held-out sequence")
        lo, hi = int(cfg["train_seed_base"]), int(cfg["train_seed_base"]) + int(cfg["episodes"])
        ensure(not any(lo <= row["seed"] < hi for row in rows), f"{name}: evaluation seeds overlap training range")
        actual = digest(folder / "final.pt")
        ensure(all(row["checkpoint"] == actual for row in rows), f"{name}: evaluation rows refer to another checkpoint")
        if cfg["game"] == "cartpole":
            ensure(all(row["length"] <= 500 and row["reward"] == row["length"] for row in rows), f"{name}: CartPole score/length incompatible with full CartPole-v1 episodes")
        else:
            ensure(all(row["reward"].is_integer() and -21 <= row["reward"] <= 21 for row in rows), f"{name}: Pong complete-episode score outside allowed range")
        stats = episode_statistics(rows)
        stats.update(checkpoint_sha256=actual, seed_first=rows[0]["seed"], seed_last=rows[-1]["seed"],
                     sampling=cfg["action_sampling"], all_rows_terminated_or_truncated=True)
        self.runs.setdefault(name, {})["evaluation"] = stats
        return stats

    def summaries(self, name):
        folder = self.root / "runs" / name
        actual = episode_statistics(read_episode_csv(folder / "evaluation.csv"))
        saved = read_json(folder / "evaluation_summary.json")
        metrics_all = read_json(self.root / "data/metrics_summary.json")
        ensure(metrics_all["mode"] == "final" and metrics_all["complete"] is True, "Global metrics summary is incomplete or from progress mode")
        metrics = metrics_all["runs"][name]
        for source_name, source in [("evaluation_summary.json", saved), ("metrics_summary.json", metrics["evaluation"])]:
            for key, expected in actual.items():
                close(source[key], expected, f"{name}/{source_name}/{key}")
            ensure(source["checkpoint_sha256"] == digest(folder / "final.pt"), f"{name}/{source_name}: checkpoint SHA mismatch")
        train = read_episode_csv(folder / "train.csv")
        train_metrics = metrics["training"]
        ensure(int(train_metrics["episodes"]) == len(train), f"{name}: saved training summary count mismatch")
        close(train_metrics["last100_mean_reward"], statistics.mean(row["reward"] for row in train[-100:]), f"{name} last100 mean")
        close(train_metrics["env_steps"], sum(row["length"] for row in train), f"{name} training steps")
        close(train_metrics["elapsed_seconds"], train[-1]["elapsed_seconds"], f"{name} training elapsed")
        return {"statistics_recomputed_from_raw_csv": actual, "sample_std_ddof": 1,
                "per_run_and_global_summary_match": True}

    def figures(self):
        from PIL import Image
        results = []
        for name in FIGURES:
            path = self.root / "figures" / name
            with Image.open(path) as image:
                ensure(image.format == "PNG", f"{name}: not a PNG")
                dimensions = image.size
                image.verify()
            results.append({"file": "figures/" + name, "dimensions": dimensions, "sha256": digest(path)})
        return {"required_contents": 11, "images": results,
                "limit": "Image integrity is automated; visual correctness is covered by the recorded PDF/page review."}

    def shuttle(self):
        import numpy as np
        cfg = read_json(self.root / "data/shuttle/config.json")
        for key, expected in {"capacity": 15, "dispatch_cost": 100.0, "holding_cost": 2.0,
                              "max_queue": 200, "gamma": 0.95, "horizon": 500, "arrivals": [1, 2, 3, 4, 5]}.items():
            ensure(cfg[key] == expected, f"Shuttle parameter {key} differs from assignment")
        saved = np.load(self.root / "data/shuttle/results.npz")
        ensure(saved["finite_values"].shape == (501, 201), "Missing complete T=500 finite-horizon values")
        ensure(np.array_equal(saved["finite_values"][-1], np.zeros(201)), "Finite terminal cost is not zero")
        value = saved["pi_value"]
        q = np.empty((2, 201))
        for action in [0, 1]:
            for state in range(201):
                left = max(0, state - 15 * action)
                q[action, state] = 100 * action + 2 * left + .95 * sum(float(value[min(200, left + a)]) for a in [1, 2, 3, 4, 5]) / 5
        residual = float(np.max(np.abs(q.min(axis=0) - value)))
        ensure(residual < 1e-8, "Saved shuttle PI is not Bellman-optimal")
        ensure(np.array_equal(q.argmin(axis=0), saved["pi_policy"]), "Saved shuttle policy is not greedy for its value")
        validation = read_json(self.root / "data/shuttle/validation.json")
        ensure(validation["passed"] is True, "Shuttle validation did not pass")
        for test in validation["tests"]:
            ensure(all(key in test and test[key] is not None for key in ["purpose", "theoretical_expectation", "method", "result", "interpretation"]), "Shuttle validation lacks a required explanation/evidence field")
        return {"bellman_residual_independently_recomputed": residual,
                "quantitative_validation_experiments": len(validation["tests"]), "finite_horizon": 500}

    def rl_validation(self):
        result = read_json(self.root / "data/validation/rl.json")
        ensure(result["status"] == "passed", "Policy-gradient correctness validation did not pass")
        for key in ["returns", "exact_gradient_and_baseline", "same_batch_mean_bias", "autograd", "cartpole_trace", "production_trainer"]:
            ensure(key in result, f"Missing RL verification evidence {key}")
        return {"status": result["status"], "evidence_sections": list(result),
                "limits": "Quantitative algorithm tests are retained; the checker does not replace scientific interpretation."}

    def repository_identity(self):
        repo = read_json(self.root / "data/repository.json")
        for key in ["authenticated_login", "repo", "url"]:
            ensure(isinstance(repo.get(key), str) and bool(repo[key].strip()), f"Missing verified repository field {key}")
        parsed = urlparse(repo["url"])
        ensure(parsed.scheme == "https" and parsed.netloc == "github.com", "Repository URL is not an actual https GitHub URL")
        parts = parsed.path.strip("/").split("/")
        ensure(len(parts) == 2 and parts[-1].removesuffix(".git") == repo["repo"], "Repository URL/name disagree")
        ensure(bool(repo.get("verification")), "No retained authentication/repository verification method")
        self.repository = repo
        return {key: repo.get(key) for key in ["authenticated_login", "repo", "url", "verification", "push_authorized"]}

    def report(self):
        from pypdf import PdfReader
        path = self.root / "output/pdf/assignment1_report.pdf"
        reader = PdfReader(str(path))
        ensure(len(reader.pages) > 0, "Final report has no pages")
        page_text = [page.extract_text() or "" for page in reader.pages]
        ensure(all(text.strip() for text in page_text), "Final report contains an empty-text page requiring inspection")
        text = "\n".join(page_text)
        markdown = (self.root / "docs/report.md").read_text()
        repo = read_json(self.root / "data/repository.json")
        ensure("Yujun Sun" in text and "Yujun Sun" in str(reader.metadata.author), "Report text/PDF metadata missing author Yujun Sun")
        for identity in [repo["authenticated_login"], repo["repo"], repo["url"]]:
            ensure(identity in text and identity in markdown, f"Report PDF/editable source missing verified identity {identity}")
        for pattern in PLACEHOLDERS:
            ensure(re.search(pattern, text + "\n" + markdown, flags=re.I) is None, f"Unfilled template marker matching {pattern}")
        for filename in FIGURES:
            ensure(filename in markdown, f"Editable report omits figure reference {filename}")
        manifest = read_json(self.root / "output/pdf/report_manifest.json")
        ensure(manifest["report_sha256"] == digest(path), "Report manifest checksum does not match current PDF")
        for name in RUNS:
            ensure(manifest["runs"][name]["checkpoint_sha256"] == digest(self.root / "runs" / name / "final.pt"), f"Report manifest refers to another {name} checkpoint")
            ensure(manifest["runs"][name]["eval_episodes"] == 500, f"Report manifest has an incomplete {name} evaluation")
        self.report_pages = len(reader.pages)
        return {"pages": len(reader.pages), "sha256": digest(path), "author": reader.metadata.author,
                "github_link_in_text_and_source": True, "placeholder_template_markers_found": False,
                "required_figure_references": 11}

    def report_visual_qa(self):
        from pypdf import PdfReader
        pdf = self.root / "output/pdf/assignment1_report.pdf"
        pages = len(PdfReader(str(pdf)).pages)
        qa = read_json(self.root / "data/report_visual_qa.json")
        ensure(qa["report_sha256"] == digest(pdf), "Per-page visual QA is for an older PDF")
        ensure(qa["passed"] is True, "Per-page visual QA has unresolved issues")
        ensure(sorted(qa["pages_examined"]) == list(range(1, pages + 1)), "Not every current report page has recorded visual inspection")
        ensure(bool(qa.get("notes")), "Per-page QA notes are missing")
        return qa

    def remote_delivery(self):
        repo = read_json(self.root / "data/repository.json")
        ensure(repo.get("remote_code_delivered", False) is True,
               "Remote code delivery remains pending; no push is implied by local completion. See data/repository.json.")
        return {"remote_code_delivered": True, "repository": repo["url"],
                "basis": "Explicit persisted delivery record; this checker neither pushes nor contacts GitHub."}

    def archive(self):
        path = self.root / "output/assignment1_Yujun_Sun.zip"
        with zipfile.ZipFile(path) as archive:
            ensure(archive.testzip() is None, "Final archive has a CRC failure")
            names = set(archive.namelist())
            ensure("MANIFEST.json" in names and "output/pdf/assignment1_report.pdf" in names, "Archive lacks manifest/report")
            for name in names:
                parts = Path(name).parts
                ensure(not any(part in {".venv", ".python", ".uv-cache", "__pycache__", ".git"} for part in parts), f"Archive includes excluded cache/environment {name}")
            for run in RUNS:
                for filename in ["final.pt", "train.csv", "evaluation.csv", "config.json", "FROZEN.json"]:
                    member = f"runs/{run}/{filename}"
                    ensure(member in names, f"Archive lacks {member}")
                    ensure(hashlib.sha256(archive.read(member)).hexdigest() == digest(self.root / member), f"Archive contains an older {member}")
        return {"path": str(path.relative_to(self.root)), "required_payload_members_verified": True,
                "crc_verified": True, "archive_is_informational_for_local_data_gate": True,
                "snapshot_note": "Required frozen models and raw experiment records were matched to current files. Acceptance records are then repackaged; no self-referential archive checksum is stored inside the archive.",
                "rebuild_after_audit": "Run scripts/package.py after this audit to include the latest acceptance records."}

    def run(self):
        self.add("assignment_source", "Read and retain the complete four-page assignment source", ["mlds_assignment1.pdf", "docs/assignment_extracted.txt"], self.source_pdf)
        self.add("required_files", "Code, exact dependencies, recovery/setup commands, English source and Chinese study guide", ["README.md", "requirements-lock.txt", "docs/LEARNING_ZH.md", "docs/report.md", "src/", "scripts/"], self.document_files)
        self.add("local_runtime", "Actual native arm64 Mac M4 Pro, 24 GB, required image/action environments", ["data/hardware.json", "data/environment.json"], self.runtime)
        self.add("device_benchmark", "Actual end-to-end CPU/MPS benchmark and measured device decision", ["data/benchmark/decision.json"], self.benchmark)
        self.add("formal_plan", "Four explicit formal training budgets recorded after measurement", ["data/experiment_plan.json", "docs/experiment_budget.md"], self.plan)
        self.add("shuttle", "All three shuttle solutions, saved values and quantitative independent validations", ["data/shuttle/", "docs/shuttle_method.md"], self.shuttle)
        self.add("rl_validation", "REINFORCE and baseline quantitative algorithm/edge/resume validations", ["data/validation/rl.json", "docs/rl_validation.md"], self.rl_validation)
        for name in RUNS:
            for suffix, title, evidence, method in [
                ("config", "Configuration matches the formal plan", ["config.json", "data/experiment_plan.json"], self.run_config),
                ("training", "Actual complete training episodes reach the formal budget", ["train.csv"], self.training),
                ("checkpoint", "Frozen final checkpoint, embedded config and raw steps agree", ["final.pt", "FROZEN.json"], self.checkpoint),
                ("evaluation", "500 complete held-out episodes with full IDs/seeds/step/checkpoint checks", ["evaluation.csv"], self.evaluation),
                ("statistics", "Saved statistics match independent raw-record recomputation", ["evaluation_summary.json", "data/metrics_summary.json"], self.summaries),
            ]:
                paths = [path if path.startswith("data/") else f"runs/{name}/{path}" for path in evidence]
                self.add(f"{name}_{suffix}", f"{name}: {title}", paths, lambda name=name, method=method: method(name))
        self.add("figures", "All eleven required plot contents exist as valid saved PNGs", ["figures/" + name for name in FIGURES], self.figures)
        self.add("repository_identity", "Verified real authenticated GitHub login, repository and URL retained", ["data/repository.json"], self.repository_identity)
        self.add("report", "Final English PDF, author, GitHub link, source, figures and checkpoint manifest", ["output/pdf/assignment1_report.pdf", "docs/report.md", "output/pdf/report_manifest.json"], self.report)
        self.add("report_visual_qa", "Every page of the current PDF visually checked for layout/numbers/formulas/links", ["data/report_visual_qa.json"], self.report_visual_qa)
        self.add("remote_delivery", "Code delivered to the real specified repository with authorization", ["data/repository.json"], self.remote_delivery,
                 required_local=False, scope="remote")
        self.add("archive", "Final work archive contains current data/checkpoints and excludes caches", ["output/assignment1_Yujun_Sun.zip"], self.archive,
                 required_local=False, scope="package")
        local_passed = all(check["passed"] for check in self.checks if check["required_local"])
        remote_passed = next(check["passed"] for check in self.checks if check["id"] == "remote_delivery")
        archive_passed = next(check["passed"] for check in self.checks if check["id"] == "archive")
        return {
            "schema_version": 1, "generated_unix_seconds": time.time(),
            "local_data_complete": local_passed, "remote_code_delivered": remote_passed,
            "archive_verified": archive_passed,
            "entire_assignment_delivered": local_passed and remote_passed and archive_passed,
            "require_remote": self.require_remote,
            "gate_passed": local_passed and (remote_passed or not self.require_remote),
            "note": "Default gate covers local scientific/report evidence. Remote delivery and archive are separately explicit; Canvas submission remains with the user.",
            "checks": self.checks, "runs": self.runs,
            "failed_required_local": [check["id"] for check in self.checks if check["required_local"] and not check["passed"]],
            "pending_remote": [] if remote_passed else ["Obtain/retain user push authorization, deliver to the verified repository, then record remote_code_delivered=true only after success."],
            "reproduction_command": ".venv/bin/python scripts/check_delivery.py" + (" --require-remote" if self.require_remote else ""),
        }


def markdown(result: dict) -> str:
    status = "PASS" if result["local_data_complete"] else "FAIL / INCOMPLETE"
    lines = ["# Assignment acceptance and evidence", "", f"Local scientific/report evidence: **{status}**.", "",
             f"Remote code delivered: **{'PASS' if result['remote_code_delivered'] else 'PENDING'}**. "
             f"Archive checked: **{'PASS' if result['archive_verified'] else 'PENDING'}**.", "",
             "This is an evidence audit of actual saved records. A passing local gate does not declare remote delivery or Canvas submission complete.", "",
             "| Requirement | Status | Evidence | Quantitative result or remaining issue |", "|---|---|---|---|"]
    for check in result["checks"]:
        tag = "PASS" if check["passed"] else ("PENDING" if check["scope"] in {"remote", "package"} else "FAIL")
        evidence = "; ".join(f"`{path}`" for path in check["evidence"])
        detail = json.dumps(check["details"], ensure_ascii=False) if check["passed"] else check["error"]
        # Full numerical records stay in JSON; keep the human checklist scannable.
        if len(detail) > 650:
            detail = detail[:647] + "... (full record in data/delivery_check.json)"
        row = [check["requirement"], tag, evidence, detail]
        lines.append("| " + " | ".join(str(item).replace("|", "\\|").replace("\n", " ") for item in row) + " |")
    lines += ["", "Reproduce this audit:", "", "```sh", result["reproduction_command"],
              ".venv/bin/python scripts/package.py", "```", "",
              "`--require-remote` additionally makes recorded remote delivery a required gate. "
              "The checker never creates a repository, pushes code, contacts GitHub, or infers authorization. "
              "Rebuild the archive after this audit so it includes these current acceptance records.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-remote", action="store_true", help="Make explicit recorded remote delivery a required gate")
    args = parser.parse_args()
    root = args.root.resolve()
    result = Audit(root, args.require_remote).run()
    (root / "data").mkdir(parents=True, exist_ok=True)
    (root / "data/delivery_check.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    (root / "ACCEPTANCE.md").write_text(markdown(result))
    print(json.dumps({key: result[key] for key in ["gate_passed", "local_data_complete", "remote_code_delivered",
                                                  "archive_verified", "entire_assignment_delivered", "failed_required_local"]}, indent=2))
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
