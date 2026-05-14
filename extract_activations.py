"""
Extract hidden-state activations from paired lie/truth datasets.

Model: specify via MODEL_REGISTRY tag or full HuggingFace model ID;
       model tag and model_id are stored in .pt metadata for traceability
Datasets: instructed_system_prompt, instructed_user_prompt, spontaneous_1, spontaneous_2,
          spontaneous_control, spontaneous_inconsistent, sycophancy_answer,
          sycophancy_are_you_sure, sycophancy_feedback, game_werewolf, game_mafia
Position: first (prefix+3), last (EOS), first_assistant (prefix+1), last_user (prefix-2),
          mean_assistant (mean over response), mid_assistant (midpoint token),
          first_k_assistant (mean first K tokens), last_k_assistant (mean last K tokens)
Output: activations_{tag}/{name}.pt (or activations_{tag}_{position}/ for non-default positions)
"""
import json
import torch
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import fire

from config import (
    ALL_DATASETS,
    DEFAULT_MODEL_TAG,
    activation_dirname,
    dataset_fingerprint,
    resolve_dataset_path,
    resolve_model,
    sample_ids,
)


def build_conversation(sample):
    conv = [
        {"role": "system", "content": sample["system_prompt"]},
        {"role": "user", "content": sample["user_message"]},
    ]
    if "followup_message" in sample:
        conv.append({"role": "assistant", "content": sample["model_response_turn1"]})
        conv.append({"role": "user", "content": sample["followup_message"]})
    conv.append({"role": "assistant", "content": sample["model_response"]})
    return conv


def get_position(position, prefix_len, seq_len, k=10):
    if position == "first":
        return min(prefix_len + 3, seq_len - 1)
    if position == "last":
        return seq_len - 1
    if position == "first_assistant":
        return min(prefix_len + 1, seq_len - 1)
    if position == "last_user":
        return max(prefix_len - 2, 0)
    if position == "mid_assistant":
        return min((prefix_len + seq_len - 1) // 2, seq_len - 1)

    resp_start = min(prefix_len, seq_len - 1)
    resp_end = seq_len - 1
    if position == "mean_assistant":
        start, end = resp_start, resp_end
    elif position == "first_k_assistant":
        start, end = resp_start, min(resp_start + k, resp_end)
    elif position == "last_k_assistant":
        start, end = max(resp_start, resp_end - k), resp_end
    else:
        raise ValueError(f"unknown position: {position}")
    if start >= end:
        return min(start, seq_len - 1)
    return (start, end)


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    return model, tokenizer


def input_device(model):
    return next(model.parameters()).device


def extract_with_model(model, tokenizer, data_dir=".", output_dir=None, position="first", max_length=2048, datasets=None, max_samples=None, k=10, model_tag="", model_id=""):
    if output_dir is None:
        output_dir = activation_dirname(model_tag, position)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if datasets:
        keep = set(datasets) if isinstance(datasets, (list, tuple)) else {d.strip() for d in datasets.split(",")}
        run_datasets = {kk: v for kk, v in ALL_DATASETS.items() if kk in keep}
    else:
        run_datasets = ALL_DATASETS

    for name, (filename, label_map) in run_datasets.items():
        data_path = resolve_dataset_path(data_dir, filename, model_tag)
        if not data_path.exists():
            raise FileNotFoundError(f"{name}: missing dataset {data_path}")
        data = json.load(open(data_path))
        if max_samples:
            data = data[:max_samples]
        labels = np.array([label_map[s["condition"]] for s in data])
        ids = sample_ids(data)
        data_hash = dataset_fingerprint(data)

        all_hidden = []
        device = input_device(model)
        for s in tqdm(data, desc=f"{name} ({position})"):
            conv = build_conversation(s)
            prefix_ids = tokenizer.apply_chat_template(conv[:-1], tokenize=True, add_generation_prompt=True)
            prefix_len = len(prefix_ids)

            text = tokenizer.apply_chat_template(conv, tokenize=False)
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            inputs = {kk: v.to(device) for kk, v in inputs.items()}
            seq_len = inputs["input_ids"].shape[1]

            pos = get_position(position, prefix_len, seq_len, k)

            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)

            n_layers = len(out.hidden_states) - 1
            if isinstance(pos, tuple):
                start, end = pos
                hidden = torch.stack([out.hidden_states[l + 1][0, start:end].mean(dim=0).cpu() for l in range(n_layers)])
            else:
                hidden = torch.stack([out.hidden_states[l + 1][0, pos].cpu() for l in range(n_layers)])
            all_hidden.append(hidden.float().numpy())

        all_hidden = np.stack(all_hidden)
        torch.save(
            {
                "activations": all_hidden,
                "labels": labels,
                "label_map": label_map,
                "model_tag": model_tag,
                "model_id": model_id,
                "dataset_name": name,
                "dataset_file": data_path.name,
                "dataset_hash": data_hash,
                "sample_ids": ids,
                "position": position,
                "max_length": max_length,
            },
            Path(output_dir) / f"{name}.pt",
            pickle_protocol=4,
        )
        print(f"{name}: {all_hidden.shape}, lies={labels.sum()}/{len(labels)}, source={data_path.name}")


def extract(model=DEFAULT_MODEL_TAG, data_dir=".", output_dir=None, position="first", max_length=2048, datasets=None, max_samples=None, k=10):
    tag, model_id = resolve_model(model)
    if output_dir is None:
        output_dir = activation_dirname(tag, position)
    m, tokenizer = load_model(model_id)
    extract_with_model(m, tokenizer, data_dir, output_dir, position, max_length, datasets, max_samples, k, model_tag=tag, model_id=model_id)


if __name__ == "__main__":
    fire.Fire(extract)
