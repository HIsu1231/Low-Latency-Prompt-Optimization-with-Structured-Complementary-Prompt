import json
import time
import torch
import argparse
import os
os.environ['HF_HOME'] = '/nas/user77/workspace/models'
os.environ['HF_CACHE'] = '/nas/user77/workspace/models'
os.environ['TRANSFORMERS_CACHE'] = '/nas/user77/workspace/models'

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "PKU-Baichuan-MLSystemLab/PAS-7B"
PROMPT_TEMPLATE = """
## Background

You are an expert in enhancing user prompts, proficient in providing detailed supplements. When identifying areas in user prompts needing further elaboration, you offer precise additions to help the user understand the core intent of their question more deeply. Focus on providing general methods and strategies, not specific details.
Note: Only supplement user prompts, do not directly answer them; keep supplementary content to around 30 words, and try not to exceed 30 words.

## Task

<User prompt>:
{}
<Complementary information>:"""
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

GENERATE_KWARGS = {
    "max_new_tokens": 1024,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.6,
    "num_beams": 1,
}

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, return_dict=True).to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# 패딩 토큰이 없으면 eos 토큰으로 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 모델 설정에서 return_dict을 True로 설정
model.config.return_dict = True

def optimize_prompt(input_text: str):
    prompt = PROMPT_TEMPLATE.format(input_text)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    start = time.time()
    with torch.no_grad():
        outputs = model.generate(**inputs, **GENERATE_KWARGS)
    latency = time.time() - start
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    try:
        optimized = decoded.split('<Complementary information>:')[1].strip()
    except IndexError:
        optimized = decoded 
    return optimized, latency

def main(dataset: str, input_path: str = None, output_path: str = None):


    if input_path.endswith('.jsonl'):
        data = []
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # CausalJudgement 데이터셋의 경우 examples 배열을 사용
        if dataset == "judgement":
            data = data.get("examples", [])

    results = []
    for entry in tqdm(data, desc=f"Processing {dataset}"):
        if dataset == "arena-hard":
            inp = entry.get("conversation", [{}])[0].get("content", "")
        elif dataset == "complex":
            inp = entry.get("instruction_en", "")
        elif dataset == "judgement":
            inp = entry.get("input", "")
        elif dataset == "gsm8k":
            inp = entry.get("question", "")
        else:
            instruction = entry.get("instruction", "")
            context = entry.get("context", "")
            inp = f"{instruction}\n{context}"

        try:
            optimized, latency = optimize_prompt(inp)
        except Exception as e:
            print(f"[Error] Failed to optimize prompt: {inp[:100]}... → {e}")
            continue

        if dataset == "dolly" or dataset == "self_instruct":
            results.append({
                "input": inp,
                "instruction": instruction,
                "context": context,
                "optimized_prompt": optimized,
                "latency": round(latency, 4)
            })
        elif dataset == "judgement":
            results.append({
                "input": inp,
                "optimized_prompt": optimized,
                "option": "\nOptions:\n- Yes\n- No",
                "latency": round(latency, 4)
            })
        else:
            results.append({
                "input": inp,
                "optimized_prompt": optimized,
                "latency": round(latency, 4)
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done! Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimize prompts using a pretrained LLM.")
    # parser.add_argument("--model_name", type=str, required=True, help="Model name (e.g., BPO, PAS)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., self_instruction)")
    parser.add_argument("--input_path", type=str, default=None, help="(Optional) Manually specify input JSON file")
    parser.add_argument("--output_path", type=str, default=None, help="(Optional) Manually specify output JSON path")
    args = parser.parse_args()

    # 전달된 인자 사용
    main(
        #model_name=args.model_name,
        dataset=args.dataset,
        input_path=args.input_path,
        output_path=args.output_path
    )
