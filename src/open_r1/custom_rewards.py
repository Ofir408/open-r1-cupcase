import re
import json
from openai import OpenAI

# --- Judge backend configuration ---
# Set JUDGE_BACKEND to 'openai' or 'hf' (HuggingFace)
# If 'hf', set HF_JUDGE_MODEL to your model name (e.g., 'meta-llama/Meta-Llama-3-8B-Instruct')
JUDGE_BACKEND = 'hf'  # or 'openai' or 'hf'
HF_JUDGE_MODEL = 'meta-llama/Meta-Llama-3-8B-Instruct'  # Only used if JUDGE_BACKEND == 'hf'

if JUDGE_BACKEND == 'openai':
    client = OpenAI()

# If using HuggingFace, lazy-load pipeline
hf_judge_pipeline = None


import re


def think_format_reward(completions: list[list[dict[str, str]]], **kwargs) -> list[float]:
    r"""
    Reward function that checks if the reasoning process is enclosed within `"<think>"` and `"</think>"` tags. The
    function returns a reward of 1.0 if the format is correct, otherwise 0.0.

    Args:
        completions (`list[list[dict[str, str]]]`):
            List of completions to be evaluated. Each completion must be a list of one message, i.e. a dictionary
            containing the key `"content"` with the value being the text of the completion.
        **kwargs:
            Additional keyword arguments. This function does not use them, but they are required in the function
            signature to ensure compatibility with trainers like [`GRPOTrainer`].

    Returns:
        `list[float]`:
            A list of rewards, where each reward is 1.0 if the completion matches the expected format, otherwise 0.0.

    Example:
    ```python
    >>> from trl.rewards import think_format_reward

    >>> completions = [
    ...     [{"content": "<think>\nThis is my reasoning.\n</think>\nThis is my answer."}],
    ...     [{"content": "<think>\nThis is my reasoning.\nThis is my answer."}],
    ... ]
    >>> think_format_reward(completions)
    [1.0, 0.0]
    ```
    """
    pattern = r"^<think>(?!.*<think>)(.*?)</think>.*$"
    # print(f"completions={completions}")
    # completion_contents = [completion[0]["content"] for completion in completions]
    completion_contents = [completion["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def think_format_reward_func(completions, **kwargs):
    # Returns 1.0 if <think>...</think> tags are present, else -1.0
    return [1.0 if think_format_reward(c) else -1.0 for c in completions]

def extract_solution(text):
    thoughts = re.findall(r'<think>(.*?)</think>', text, flags=re.DOTALL)
    thought = "".join(thoughts).strip()
    parts = re.split(r'</think>', text, flags=re.DOTALL)
    solution = parts[-1].strip()
    solution = re.sub(r'\s*<\|im_end\|>\s*', '', solution)
    return thought, solution


def check_diagnosis_match(final_diagnosis, generated_diagnosis):
    """Check if two diagnoses match using an LLM as a judge (OpenAI or HuggingFace)."""
    prompt = f"""
You are a medical expert acting as a judge. Your task is to determine whether the following two diagnoses refer to the same underlying medical diagnosis.

Final Diagnosis: {final_diagnosis}
Generated Diagnosis: {generated_diagnosis}

Evaluation Guidelines:
1. Do not treat merely related conditions as equivalent.
2. An exact wording match is not required—evaluate whether the core underlying medical diagnosis is the same.
3. If the predicted diagnosis is a subtype or more specific form of the correct diagnosis, consider them equivalent.

Respond with JSON in this format:
{{
  "result": "True" | "False" | "Not Sure",
  "explanation": "Brief explanation"
}}
"""
    try:
        if JUDGE_BACKEND == 'openai':
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            answer = response.choices[0].message.content.strip()
            response_json = json.loads(answer)
            return response_json.get("result", "Not Sure")
        elif JUDGE_BACKEND == 'hf':
            global hf_judge_pipeline
            if hf_judge_pipeline is None:
                from transformers import pipeline
                hf_judge_pipeline = pipeline("text-generation", model=HF_JUDGE_MODEL, max_new_tokens=256)
            # HuggingFace pipeline returns a list of dicts with 'generated_text' or 'text'
            out = hf_judge_pipeline(prompt, return_full_text=False)
            if isinstance(out, list):
                text = out[0].get('generated_text', out[0].get('text', '')).strip()
            else:
                text = str(out)
            # Try to extract JSON from the output
            try:
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    answer_json = text[json_start:json_end]
                    response_json = json.loads(answer_json)
                    return response_json.get("result", "Not Sure")
            except Exception as e:
                print(f"Failed to parse HuggingFace judge output: {e}, output: {text}")
            return "Not Sure"
        else:
            raise ValueError(f"Unknown JUDGE_BACKEND: {JUDGE_BACKEND}")
    except Exception as e:
        print(f"exception: {e}, using not sure")
        raise e
        return "Not Sure"

def diagnosis_match_reward_func(completions, references, **kwargs):
    # references: list of ground-truth answers (training_format)
    # completions: list of model outputs
    rewards = []
    for completion, reference in zip(completions, references):
        _, gt_diag = extract_solution(reference)
        _, pred_diag = extract_solution(completion)
        judge_result = check_diagnosis_match(gt_diag, pred_diag)
        if judge_result == "True":
            rewards.append(1.0)
        elif judge_result == "False":
            rewards.append(0.0)
        else:
            rewards.append(0.5)
    return rewards

def check_diagnosis_similarity(final_diagnosis, generated_diagnosis):
    """Check similarity between two diagnoses using an LLM as a judge (OpenAI or HuggingFace)."""
    prompt = f"""
You are a medical expert acting as a judge. Your task is to critically assess, and determine whether the following two diagnoses are:
1. Exact Same – Both refer to the same underlying medical diagnosis, despite possible wording differences.
2. Very Similar – The diagnoses are not identical, but are clinically very similar in presentation, management, or implications.
3. Same Mechanisms – The diagnoses are distinct and not clinically similar, but involve the same physiological systems or pathological mechanisms.
4. Different – The diagnoses are clearly distinct, with minimal overlap in clinical presentation, mechanisms, or affected systems.

Diagnoses to Assess:
- First Diagnosis: {final_diagnosis}
- Second Diagnosis: {generated_diagnosis}

Evaluation Guidelines:
- Do not consider related but distinct conditions to be equivalent.
- Focus on the underlying clinical and pathological nature of each diagnosis, not just the wording.

Response Format (JSON):
{{
  "result": "Exact Same" | "Very Similar" | "Same Mechanisms" | "Different",
  "explanation": "Brief explanation of the reasoning behind your classification"
}}
"""
    try:
        if JUDGE_BACKEND == 'openai':
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                response_format={"type": "json_object"}
            )
            answer = response.choices[0].message.content.strip()
            response_json = json.loads(answer)
            return response_json.get("result", None)
        elif JUDGE_BACKEND == 'hf':
            global hf_judge_pipeline
            if hf_judge_pipeline is None:
                from transformers import pipeline
                hf_judge_pipeline = pipeline("text-generation", model=HF_JUDGE_MODEL, max_new_tokens=256, temperature=0.4)
            out = hf_judge_pipeline(prompt, return_full_text=False)
            if isinstance(out, list):
                text = out[0].get('generated_text', out[0].get('text', '')).strip()
            else:
                text = str(out)
            try:
                print(f"judge generation={text}")
                json_start = text.find('{')
                json_end = text.rfind('}') + 1
                if json_start != -1 and json_end != -1:
                    answer_json = text[json_start:json_end]
                    response_json = json.loads(answer_json)
                    return response_json.get("result", None)
            except Exception as e:
                print(f"Failed to parse HuggingFace judge output: {e}, output: {text}")
            return None
        else:
            raise ValueError(f"Unknown JUDGE_BACKEND: {JUDGE_BACKEND}")
    except Exception as e:
        print(f"exception: {e}, using None")
        return None

def diagnosis_similarity_reward_func(completions, solution: list[str], **kwargs):
    # references: list of ground-truth answers (training_format)
    # completions: list of model outputs
    scores = {
        "Exact Same": 1.0,
        "Very Similar": 0.7,
        "Same Mechanisms": 0.2,
        "Different": -1.0
    }
    rewards = []
    # print(f"solution={solution}")
    # print(f"completions={completions}")
    contents = [completion[0]["content"] for completion in completions]
    # print(f"contents={contents}")
    
    for completion, reference in zip(contents, solution):
        _, gt_diag = extract_solution(reference)
        _, pred_diag = extract_solution(completion)
        judge_result = check_diagnosis_similarity(gt_diag, pred_diag)
        reward = scores.get(judge_result, None)
        rewards.append(reward)
    return rewards
