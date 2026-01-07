import requests
import json
import os
os.environ['HF_HOME'] = '/nas/user77/workspace/models'
os.environ['HF_CACHE'] = '/nas/user77/workspace/models'
os.environ['TRANSFORMERS_CACHE'] = '/nas/user77/workspace/models'
import time
import random
from tqdm import tqdm
import pathlib
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

#HF_TOKEN = os.getenv("HF_TOKEN")



def generate_prompt(input, dataset):

    if dataset == 'dolly' or dataset =="self_instruct":
        prompt = input['input'] + '\n' + input['context']
    elif dataset == 'judgement':
        prompt = 
    else:
        prompt = input['input']

    system_prompt_template = """As a {}, your task is to respond to an audience of {}.  
The user's goal is to {}, so your answer should be expressed in a {} tone.  
Ensure your response adheres to the following constraints: {}.  
Present your reasoning, guided by: {}.  
Your answer should follow this format: {}.  
You are required to follow the interaction rule (questioning at the end of output or not) exactly as specified: {}."""

    system_prompt = system_prompt_template.format(
        input['prediction']['Role'],
        input['prediction']['Audience'],
        input['prediction']['User Intent'],
        input['prediction']['Tone Type'],
        input['prediction']['Constraints'],
        input['prediction']['Reasoning Guidance'],
        input['prediction']['Output Format'],
        input['prediction']['Interactive Mode']
    )
    return prompt, system_prompt

def generate_supplementary_ours(input, model, tokenizer, max_retries=5, wait_base=2):
    for attempt in range(max_retries):
        try:
            prompt, system_prompt = generate_prompt(input)


            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            device = next(model.parameters()).device
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            start_time = time.time()
            generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512, temperature=0.9)
            end_time = time.time()

            # 입력 토큰 길이 계산 후 생성된 토큰만 추출
            input_length = model_inputs.input_ids.shape[1]
            generated_tokens = generated_ids[0, input_length:]
            output = tokenizer.decode(generated_tokens, skip_special_tokens=True)


            latency = round(end_time - start_time, 6)
            return output, latency, prompt, system_prompt

        except Exception as e:
            print(f"[Attempt {attempt+1}] Error: {repr(e)}")
            if attempt < max_retries - 1:
                wait_time = wait_base * (2 ** attempt)
                print(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries exceeded.\n")
                return None, None, None, None

def generate_supplementary(input, model, tokenizer, task, dataset=None, max_retries=5, wait_base=2):


    if task == 'Ori':
        prompt = input['input'] + '\n' + input['context']
    elif task == 'CoT':
        prompt = input['input'] + "\n Let's think step by step." + '\n' + input['context']
    elif task == 'PAS':
        prompt = input['input'] + '\n' + input['optimized_prompt'] + '\n' + input['context']
    elif task == 'BPO' or task == 'FIPO':
        if dataset == 'dolly' or dataset =="self_instruct":
            prompt = input['optimized_prompt'] + '\n' + input['context']
        else:
            prompt = input['optimized_prompt']
    else:
        raise ValueError(f"Invalid task: {task}")

    for attempt in range(max_retries):
        try:

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to("cuda")

            start_time = time.time()
            generated_ids = model.generate(
                model_inputs.input_ids, 
                attention_mask=model_inputs.attention_mask,
                max_new_tokens=512, 
                temperature=0.9
            )
            end_time = time.time()
            
            # 입력 토큰 길이를 계산하여 생성된 토큰만 추출
            input_length = model_inputs.input_ids.shape[1]
            generated_tokens = generated_ids[0, input_length:]
            output = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            latency = round(end_time - start_time, 6)
            return prompt, output, latency
        except Exception as e:
            print(f"[Attempt {attempt+1}] Error in generate_supplementary: {repr(e)}")
            if attempt < max_retries - 1:
                wait_time = wait_base * (2 ** attempt)
                print(f"Retrying after {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries exceeded.")
                return None, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--model_id', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 병렬 처리 비활성화

    total, success, failed = 0, 0, 0



    dataset = []
    
    # 파일 확장자에 따라 처리 방식 결정
    if args.input_path.endswith('.jsonl'):
        # JSONL 형식 처리
        with open(args.input_path, 'r', encoding='utf-8') as infile:
            try:
                for line in infile:
                    line = line.strip()
                    if line:
                        dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f'Failed to parse input JSONL: {e}')
                return
    elif args.input_path.endswith('.json'):
        # JSON 형식 처리
        try:
            with open(args.input_path, 'r', encoding='utf-8') as infile:
                data = json.load(infile)
                # JSON 파일이 리스트인 경우
                if isinstance(data, list):
                    dataset = data
                # JSON 파일이 딕셔너리인 경우, 단일 데이터로 처리
                elif isinstance(data, dict):
                    dataset = [data]
                else:
                    print(f'Unsupported JSON format: {type(data)}')
                    return
        except json.JSONDecodeError as e:
            print(f'Failed to parse input JSON: {e}')
            return
    else:
        print(f'Unsupported file format. Please use .json or .jsonl files.')
        return

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    with open(args.output_path, 'w', encoding='utf-8') as outfile:
        with torch.no_grad():
            for data in tqdm(dataset, desc='Generating Supplementary Prompt'):
                total += 1
                if args.task == 'RoBERTa':
                    output, latency, p, sys_p = generate_supplementary_ours(data, model, tokenizer)
                    if output:
                        success += 1
                        entry = {
                            'input': data['input'],
                            'complementary structure': data['prediction'],
                            'prompt': p,
                            'system_prompt': sys_p,
                            'output': output,
                            'latency': latency
                        }
                    else:
                        failed += 1
                        entry = {
                            'input': data['input'],
                            'complementary structure': data['prediction'],
                            'output': None,
                            'latency': None,
                            'error': "generation_failed"
                        }
                else:
                    prompt, output, latency = generate_supplementary(data, model, tokenizer, args.task, args.dataset)
                    if output:
                        success += 1
                        entry = {
                            'input': data['input'],
                            'prompt': prompt,
                            'output': output,
                            'latency': latency
                        }
                    else:
                        failed += 1
                        entry = {
                            'input': data['input'],
                            'prompt': prompt,
                            'output': None,
                            'latency': None,
                            'error': "generation_failed"
                        }
                outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print('\n--- Summary ---')
    print(f'Total: {total} | Success: {success} | Failed: {failed}')

if __name__ == '__main__':
    main()


#CUDA_VISIBLE_DEVICES=2 python evaluation/src/inference.py --input_path evaluation/outputs/FIPO/optimized_prompt/dolly.jsonl --output_path evaluation/outputs/FIPO/inferenced_outputs/FIPO_Qwen-2-0.5B-Instruct_outputs.jsonl --model_id Qwen/Qwen2-0.5B-Instruct --task FIPO
#CUDA_VISIBLE_DEVICES=2 python evaluation/src/inference.py --input_path evaluation/outputs/FIPO/optimized_prompt/dolly.jsonl --output_path evaluation/outputs/FIPO/inferenced_outputs/FIPO_Qwen-2-1.5B-Instruct_outputs.jsonl --model_id Qwen/Qwen2-1.5B-Instruct --task FIPO
#CUDA_VISIBLE_DEVICES=2 python evaluation/src/inference.py --input_path evaluation/outputs/FIPO/optimized_prompt/dolly.jsonl --output_path evaluation/outputs/FIPO/inferenced_outputs/FIPO_Qwen-2-7B-Instruct_outputs.jsonl --model_id Qwen/Qwen2-7B-Instruct --task FIPO