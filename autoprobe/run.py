"""Run autoprobe candidate experiments and log keep/discard decisions."""

from __future__ import annotations

import importlib
import json
import argparse
from datetime import datetime
from pathlib import Path

from config import DEFAULT_MODEL_TAG, PROBING_ROOT, resolve_model

from .evaluator import EvaluationContext, evaluate_safely, write_report
from .specs import ExperimentSpec, safe_name

RESULT_HEADER = [
    "experiment",
    "robust_score",
    "val_mean_auroc",
    "val_min_auroc",
    "length_penalty",
    "dataset_identity_penalty",
    "runtime_seconds",
    "status",
    "decision",
    "description",
    "report",
]


def _load_candidates(module_name: str, context: dict) -> list[ExperimentSpec]:
    module = importlib.import_module(module_name)
    specs = module.propose(context)
    normalized = []
    for spec in specs:
        if isinstance(spec, ExperimentSpec):
            spec.validate()
            normalized.append(spec)
        elif isinstance(spec, dict):
            normalized.append(ExperimentSpec.from_mapping(spec))
        else:
            raise TypeError(f"{module_name}.propose returned unsupported item {type(spec).__name__}")
    return normalized


def _append_result_tsv(path: Path, row: dict) -> None:
    exists = path.exists()
    with open(path, "a") as f:
        if not exists:
            f.write("\t".join(RESULT_HEADER) + "\n")
        f.write("\t".join(str(row.get(k, "")) for k in RESULT_HEADER) + "\n")


def _summary_row(spec: ExperimentSpec, result: dict, decision: str, report_path: Path) -> dict:
    return {
        "experiment": spec.name,
        "robust_score": f"{result.get('robust_score', 0.0):.6f}" if result.get("status") == "ok" else "0.000000",
        "val_mean_auroc": f"{result.get('val_mean_auroc', 0.0):.6f}" if result.get("status") == "ok" else "0.000000",
        "val_min_auroc": f"{result.get('val_min_auroc', 0.0):.6f}" if result.get("status") == "ok" else "0.000000",
        "length_penalty": f"{result.get('length_penalty', 0.0):.6f}" if result.get("status") == "ok" else "0.000000",
        "dataset_identity_penalty": f"{result.get('dataset_identity_penalty', 0.0):.6f}" if result.get("status") == "ok" else "0.000000",
        "runtime_seconds": f"{result.get('runtime_seconds', 0.0):.3f}" if result.get("status") == "ok" else "0.000",
        "status": result.get("status", "unknown"),
        "decision": decision,
        "description": spec.description.replace("\t", " "),
        "report": str(report_path),
    }


def main(
    run_tag: str | None = None,
    mode: str = "smoke",
    model: str = DEFAULT_MODEL_TAG,
    data_dir: str = ".",
    output_root: str = "autoprobe/results",
    candidate_module: str = "autoprobe.candidates",
    max_experiments: int | None = None,
    baseline_runtime_seconds: float = 60.0,
) -> None:
    model_tag, _ = resolve_model(model)
    run_tag = run_tag or datetime.now().strftime("%Y%m%d_%H%M%S")
    data_path = Path(data_dir).resolve()
    if data_path == Path(".").resolve():
        data_path = PROBING_ROOT
    out_dir = Path(output_root) / run_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    context_dict = {
        "mode": mode,
        "model_tag": model_tag,
        "data_dir": str(data_path),
        "output_dir": str(out_dir),
    }
    specs = _load_candidates(candidate_module, context_dict)
    if max_experiments is not None:
        specs = specs[:max_experiments]

    context = EvaluationContext(
        data_dir=data_path,
        model_tag=model_tag,
        mode=mode,
        baseline_runtime_seconds=baseline_runtime_seconds,
    )

    print(f"autoprobe run_tag={run_tag}")
    print(f"mode={mode}, model_tag={model_tag}, data_dir={data_path}")
    print(f"candidate_module={candidate_module}, experiments={len(specs)}")
    print(f"output={out_dir}")

    tsv_path = out_dir / "results.tsv"
    best_score = float("-inf")
    best_result = None

    for idx, spec in enumerate(specs, start=1):
        print(f"\n[{idx}/{len(specs)}] {spec.name}")
        result = evaluate_safely(spec, context)
        status = result.get("status", "unknown")
        decision = status
        if status == "ok":
            score = float(result["robust_score"])
            decision = "keep" if score > best_score else "discard"
            if score > best_score:
                best_score = score
                best_result = result
            print(
                f"  robust={score:.4f} val_mean={result['val_mean_auroc']:.4f} "
                f"val_min={result['val_min_auroc']:.4f} decision={decision}"
            )
        else:
            print(f"  {status}: {result.get('reason') or result.get('error')}")

        report_path = out_dir / f"{idx:03d}_{safe_name(spec.name)}.json"
        result["decision"] = decision
        write_report(report_path, result)
        _append_result_tsv(tsv_path, _summary_row(spec, result, decision, report_path))

    if best_result is not None:
        best_dir = out_dir / "best"
        best_dir.mkdir(exist_ok=True)
        write_report(best_dir / "best_result.json", best_result)
        with open(best_dir / "best_spec.json", "w") as f:
            json.dump(best_result["spec"], f, indent=2)
        print(f"\nbest: {best_result['spec']['name']} robust={best_result['robust_score']:.4f}")
    else:
        print("\nno successful experiments")
    print(f"results: {tsv_path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run autoprobe candidate experiments.")
    parser.add_argument("--run-tag", default=None)
    parser.add_argument("--mode", default="smoke")
    parser.add_argument("--model", default=DEFAULT_MODEL_TAG)
    parser.add_argument("--data-dir", default=".")
    parser.add_argument("--output-root", default="autoprobe/results")
    parser.add_argument("--candidate-module", default="autoprobe.candidates")
    parser.add_argument("--max-experiments", type=int, default=None)
    parser.add_argument("--baseline-runtime-seconds", type=float, default=60.0)
    return parser


if __name__ == "__main__":
    args = _parser().parse_args()
    main(
        run_tag=args.run_tag,
        mode=args.mode,
        model=args.model,
        data_dir=args.data_dir,
        output_root=args.output_root,
        candidate_module=args.candidate_module,
        max_experiments=args.max_experiments,
        baseline_runtime_seconds=args.baseline_runtime_seconds,
    )
