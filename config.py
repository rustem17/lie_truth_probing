import hashlib
import json
import re
from pathlib import Path

PROBING_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROBING_ROOT

TRAIN_DATASETS = {
    "instructed_system_prompt": ("instructed_system_prompt.json", {"lie": 1, "truth": 0}),
    "spontaneous_1": ("spontaneous_1.json", {"spontaneous_lie": 1, "spontaneous_truth": 0}),
    "sycophancy_answer": ("sycophancy_answer.json", {"sycophantic_lie": 1, "sycophantic_truth": 0}),
    "sycophancy_feedback": ("sycophancy_feedback.json", {"sycophantic_lie": 1, "sycophantic_truth": 0}),
    "game_werewolf": ("game_werewolf.json", {"game_lie": 1, "game_truth": 0}),
    "spontaneous_inconsistent": ("spontaneous_inconsistent.json", {"spontaneous_lie": 1, "spontaneous_truth": 0}),
}

VALIDATION_DATASETS = {
    "instructed_user_prompt": ("instructed_user_prompt.json", {"lie": 1, "truth": 0}),
    "spontaneous_2": ("spontaneous_2.json", {"spontaneous_lie": 1, "spontaneous_truth": 0}),
    "spontaneous_control": ("spontaneous_control.json", {"spontaneous_lie": 1, "spontaneous_truth": 0}),
    "sycophancy_are_you_sure": ("sycophancy_are_you_sure.json", {"sycophantic_lie": 1, "sycophantic_truth": 0}),
    "game_mafia": ("game_mafia.json", {"game_lie": 1, "game_truth": 0}),
}

VALIDATION_MAP = {
    "instructed_system_prompt": [
        ("instructed_user_prompt", "instructed_user_prompt.json", {"lie": 1, "truth": 0}),
    ],
    "spontaneous_1": [
        ("spontaneous_2", "spontaneous_2.json", {"spontaneous_lie": 1, "spontaneous_truth": 0}),
        ("spontaneous_control", "spontaneous_control.json", {"spontaneous_lie": 1, "spontaneous_truth": 0}),
    ],
    "sycophancy_answer": [
        ("sycophancy_are_you_sure", "sycophancy_are_you_sure.json", {"sycophantic_lie": 1, "sycophantic_truth": 0}),
        ("sycophancy_feedback", "sycophancy_feedback.json", {"sycophantic_lie": 1, "sycophantic_truth": 0}),
    ],
    "game_werewolf": [
        ("game_mafia", "game_mafia.json", {"game_lie": 1, "game_truth": 0}),
    ],
}

ALL_DATASETS = {**TRAIN_DATASETS, **VALIDATION_DATASETS}

COLORS = {
    "instructed_system_prompt": "#1F77B4",
    "instructed_user_prompt": "#AEC7E8",
    "spontaneous_1": "#2CA02C",
    "spontaneous_2": "#98DF8A",
    "spontaneous_control": "#C5B0D5",
    "spontaneous_inconsistent": "#9467BD",
    "sycophancy_answer": "#FF7F0E",
    "sycophancy_are_you_sure": "#FFBB78",
    "sycophancy_feedback": "#C49C94",
    "game_werewolf": "#D62728",
    "game_mafia": "#FF9896",
}

SHORT = {
    "instructed_system_prompt": "inst_sys",
    "instructed_user_prompt": "inst_user",
    "spontaneous_1": "spont_1",
    "spontaneous_2": "spont_2",
    "spontaneous_control": "spont_ctrl",
    "spontaneous_inconsistent": "spont_inc",
    "sycophancy_answer": "syco_ans",
    "sycophancy_are_you_sure": "syco_ays",
    "sycophancy_feedback": "syco_fb",
    "game_werewolf": "werewolf",
    "game_mafia": "mafia",
}

MARKERS = {
    "instructed_system_prompt": "o",
    "instructed_user_prompt": "P",
    "spontaneous_1": "s",
    "spontaneous_2": "X",
    "spontaneous_control": "p",
    "spontaneous_inconsistent": "v",
    "sycophancy_answer": "D",
    "sycophancy_are_you_sure": "h",
    "sycophancy_feedback": "d",
    "game_werewolf": "^",
    "game_mafia": "*",
}

_PAIR_PALETTE = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
                 "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF"]
_PAIR_COLOR_CACHE = {}

def pair_color(a, b):
    key = tuple(sorted([a, b]))
    if key not in _PAIR_COLOR_CACHE:
        _PAIR_COLOR_CACHE[key] = _PAIR_PALETTE[len(_PAIR_COLOR_CACHE) % len(_PAIR_PALETTE)]
    return _PAIR_COLOR_CACHE[key]


DEFAULT_MODEL_TAG = "llama-3-3-70b-instruct"
ACTIVATIONS_DIR = PROBING_ROOT / f"activations_{DEFAULT_MODEL_TAG}"

MODEL_REGISTRY = {
    "llama-3-3-70b-instruct": "meta-llama/Llama-3.3-70B-Instruct",
    "gemma-4-31b-it": "google/gemma-4-31B-it",
    "olmo-2-0325-32b": "allenai/OLMo-2-0325-32B",
    "qwen3-6-35b-a3b": "Qwen/Qwen3.6-35B-A3B",
    "glm-5-1": "zai-org/GLM-5.1",
    "trinity-large-thinking": "arcee-ai/Trinity-Large-Thinking",
    "minimax-m2-7": "MiniMaxAI/MiniMax-M2.7",
    "kimi-linear-48b-a3b-instruct": "moonshotai/Kimi-Linear-48B-A3B-Instruct",
}


def sanitize_model_tag(model_id):
    name = str(model_id).split("/")[-1]
    return re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-").lower()


def resolve_model(tag_or_id=None):
    tag_or_id = tag_or_id or DEFAULT_MODEL_TAG
    if tag_or_id in MODEL_REGISTRY:
        return tag_or_id, MODEL_REGISTRY[tag_or_id]
    for tag, mid in MODEL_REGISTRY.items():
        if mid == tag_or_id:
            return tag, mid
    return sanitize_model_tag(tag_or_id), tag_or_id


def tagged_filename(base_name, model_tag=""):
    p = Path(base_name)
    return f"{p.stem}_{model_tag}{p.suffix}" if model_tag else p.name


def dataset_filename(base_name, model_tag=""):
    return tagged_filename(base_name, model_tag)


def infer_output_filename(base_name, model_tag=""):
    return tagged_filename(base_name, model_tag)


def activation_dirname(model_tag="", position="first"):
    if not model_tag:
        return "activations" if position == "first" else f"activations_{position}"
    return f"activations_{model_tag}" if position == "first" else f"activations_{model_tag}_{position}"


def resolve_dataset_path(data_dir, base_name, model_tag="", prefer_tagged=True):
    data_dir = Path(data_dir)
    candidates = []
    if model_tag and prefer_tagged:
        candidates.append(data_dir / tagged_filename(base_name, model_tag))
    candidates.append(data_dir / base_name)
    if model_tag and not prefer_tagged:
        candidates.append(data_dir / tagged_filename(base_name, model_tag))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_dataset_path_for_activation(data_dir, base_name, model_tag="", activation_meta=None):
    activation_meta = activation_meta or {}
    saved_file = activation_meta.get("dataset_file")
    if saved_file:
        saved_path = Path(data_dir) / saved_file
        if saved_path.exists():
            return saved_path
    return resolve_dataset_path(data_dir, base_name, model_tag)


def dataset_fingerprint(samples):
    payload = json.dumps(samples, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sample_ids(samples):
    return [s.get("id") for s in samples]


def validate_dataset_provenance(activation_meta, samples, dataset_name):
    saved_ids = activation_meta.get("sample_ids")
    current_ids = sample_ids(samples)
    if saved_ids is not None and saved_ids != current_ids:
        raise ValueError(
            f"{dataset_name}: sample ids do not match activation provenance. "
            "Regenerate activations or use the original dataset file."
        )

    saved_hash = activation_meta.get("dataset_hash")
    if saved_hash is not None:
        current_hash = dataset_fingerprint(samples)
        if saved_hash != current_hash:
            raise ValueError(
                f"{dataset_name}: dataset hash does not match activation provenance. "
                "Regenerate activations or use the original dataset file."
            )
