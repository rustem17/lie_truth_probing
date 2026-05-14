"""Multi-run vLLM inference for spontaneous MCQ dataset.

Model (inference): vLLM-served; pass --model=<tag> (see config.MODEL_REGISTRY) or full HF ID
Model (judge): Anthropic API (default claude-haiku-4-5-20251001), only for regex failures
Input: spontaneous_matched_dataset.json (2000 MCQ samples)
Output: multi_results_{model_tag}.json — checkpoint with parsed_answers per run, LLM-resolved where regex failed
Temperature: 0.6, N_runs: 10
"""
import asyncio
import json
import re
from pathlib import Path
from collections import Counter

import fire
from dotenv import load_dotenv
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tqdm.asyncio import tqdm_asyncio

load_dotenv()

MAX_CONCURRENT_INFER = 32
MAX_CONCURRENT_JUDGE = 32
infer_sem = asyncio.Semaphore(MAX_CONCURRENT_INFER)
judge_sem = asyncio.Semaphore(MAX_CONCURRENT_JUDGE)

ANSWER_PATTERNS = [
    re.compile(r"(?:the\s+)?(?:correct\s+)?answer\s+is\s*:?\s*\(?([A-J])\)?", re.IGNORECASE),
    re.compile(r"\b([A-J])\.\s", re.MULTILINE),
    re.compile(r"\(([A-J])\)\s*$", re.MULTILINE),
]
FALLBACK = re.compile(r"(?<![a-zA-Z0-9*^/\\])([A-J])(?:\.|(?:\s*[\)\.])|$)")

JUDGE_PROMPT = """A model answered a multiple-choice question. What answer letter (A-J) did it select?

Question:
{question}

Model's response:
{response}

Respond with ONLY the letter (A-J) or "unclear". Nothing else."""


def atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def load_existing(output):
    path = Path(output)
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {s["id"]: s for s in data if "id" in s}


def checkpoint_rows(dataset, all_responses, requested_runs=None):
    return [
        {**s, "model_responses": all_responses[i],
         "n_runs": requested_runs or len(all_responses[i])}
        for i, s in enumerate(dataset)
    ]


def extract_answer(response):
    for pat in ANSWER_PATTERNS:
        matches = pat.findall(response)
        if matches:
            return matches[-1].upper()
    matches = FALLBACK.findall(response)
    if matches:
        return matches[-1].upper()
    return None


async def sample(client, system_prompt, user_message, model, temperature):
    async with infer_sem:
        r = await client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_message}],
            temperature=temperature, max_tokens=1024)
        return r.choices[0].message.content


async def judge_one(client, question, response, model):
    async with judge_sem:
        r = await client.messages.create(
            model=model, max_tokens=8,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, response=response[:800])}])
        text = r.content[0].text.strip().upper()
        if len(text) == 1 and text in "ABCDEFGHIJ":
            return text
        return None


async def run(model, base_url, dataset_path, output, n_runs, temperature, judge_model):
    with open(dataset_path) as f:
        dataset = json.load(f)

    existing = load_existing(output)
    infer_client = AsyncOpenAI(api_key="not-needed", base_url=base_url)
    all_responses = {
        i: list(existing.get(s["id"], {}).get("model_responses", []))[:n_runs]
        for i, s in enumerate(dataset)
    }
    for run_idx in range(n_runs):
        missing = [i for i in range(len(dataset)) if len(all_responses[i]) <= run_idx]
        if not missing:
            print(f"Run {run_idx + 1}/{n_runs}: already complete")
            continue
        print(f"Run {run_idx + 1}/{n_runs}: {len(missing)} missing")
        tasks = [sample(infer_client, dataset[i]["system_prompt"], dataset[i]["user_message"], model, temperature)
                 for i in missing]
        responses = await tqdm_asyncio.gather(*tasks)
        for i, resp in zip(missing, responses):
            all_responses[i].append(resp)
        atomic_write_json(output, checkpoint_rows(dataset, all_responses, requested_runs=n_runs))

    all_answers = {}
    to_judge = []
    for i, s in enumerate(dataset):
        old_answers = list(existing.get(s["id"], {}).get("parsed_answers", []))[:n_runs]
        answers = []
        for j, resp in enumerate(all_responses[i]):
            if j < len(old_answers) and old_answers[j] is not None:
                a = old_answers[j]
            else:
                a = extract_answer(resp)
            if a is None:
                to_judge.append((i, j, s["user_message"], resp))
            answers.append(a)
        all_answers[i] = answers

    if to_judge:
        print(f"Regex failed on {len(to_judge)} responses, judging with LLM...")
        judge_client = AsyncAnthropic()
        judge_tasks = [judge_one(judge_client, q, r, judge_model) for _, _, q, r in to_judge]
        judge_results = await tqdm_asyncio.gather(*judge_tasks)
        resolved = sum(1 for r in judge_results if r is not None)
        print(f"Judge resolved: {resolved}/{len(to_judge)}")
        for (i, j, _, _), new_a in zip(to_judge, judge_results):
            all_answers[i][j] = new_a

    results = []
    consistent = 0
    for i, s in enumerate(dataset):
        answers = all_answers[i]
        counts = Counter(a for a in answers if a)
        majority_answer, majority_count = counts.most_common(1)[0] if counts else (None, 0)
        is_consistent = majority_count == n_runs
        if is_consistent:
            consistent += 1
        results.append({
            **s,
            "model_responses": all_responses[i],
            "parsed_answers": answers,
            "parsed_answer": majority_answer,
            "answer_counts": dict(counts),
            "n_runs": n_runs,
            "consistent": is_consistent,
        })

    atomic_write_json(output, results)

    print(f"\n{len(results)} samples, {consistent} consistent "
          f"({100*consistent/len(results):.1f}%) -> {output}")


def main(model=None, base_url="http://localhost:8000/v1",
         dataset="spontaneous_matched_dataset.json", output=None, model_tag="",
         n_runs=10, temperature=0.6, judge_model="claude-haiku-4-5-20251001"):
    import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from config import DEFAULT_MODEL_TAG, resolve_model
    model = model or DEFAULT_MODEL_TAG
    model_tag, model = resolve_model(model) if not model_tag else (model_tag, resolve_model(model)[1])
    if not output:
        output = f"multi_results_{model_tag}.json" if model_tag else "multi_results.json"
    asyncio.run(run(model, base_url, dataset, output, n_runs, temperature, judge_model))


if __name__ == "__main__":
    fire.Fire(main)
