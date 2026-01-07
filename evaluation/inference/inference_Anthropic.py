import json
import os
from dotenv import load_dotenv
import anthropic
# .env 파일 로드
load_dotenv()
os.environ['HF_HOME'] = '/nas/user77/workspace/models'
os.environ['HF_CACHE'] = '/nas/user77/workspace/models'
os.environ['TRANSFORMERS_CACHE'] = '/nas/user77/workspace/models'
import time
import random
from tqdm import tqdm
import pathlib
import argparse

# Anthropic API 키 설정
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    print("Warning: ANTHROPIC_API_KEY not found in environment variables")
    print("Please create a .env file in evaluation/src/ directory with:")
    print("ANTHROPIC_API_KEY=your_actual_api_key_here")
    exit(1)

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

def generate_prompt(input):
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

def generate_supplementary_ours(input, model_name,dataset, max_retries=6, wait_base=2, max_wait=60):
    for attempt in range(max_retries):
        try:
            prompt, system_prompt = generate_prompt(input)

            if dataset == 'judgement':
                prompt = prompt + "\nOptions:\n- Yes\n- No"
            else:
                prompt = prompt
            
            start_time = time.time()
            response = client.messages.create(
                model=model_name,
                max_tokens=512,
                temperature=0.9,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            end_time = time.time()

            output = response.content[0].text
            latency = round(end_time - start_time, 6)
            return output, latency, prompt, system_prompt

        except Exception as e:
            print(f"[Attempt {attempt + 1}/{max_retries}] Error: {repr(e)}")
            
            if attempt < max_retries - 1:  # 마지막 시도가 아닌 경우에만 대기
                # 백오프 전략: 지수적으로 증가하되 최대 대기 시간 제한
                wait_time = min(wait_base * (2 ** attempt), max_wait)
                # 약간의 랜덤성 추가하여 동시 요청 시 충돌 방지
                wait_time = wait_time + random.uniform(0, wait_time * 0.1)
                
                print(f"Retrying after {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries exceeded.")
                return None, None, None, None

def generate_supplementary(input, model_name, task, dataset, max_retries=6, wait_base=2, max_wait=60):
    
    if task == 'Ori':
        if dataset == 'judgement':
            prompt = input['input'] + input['option']
        else:
            prompt = input['input']
    elif task == 'CoT':
        if dataset == 'judgement':
            prompt = input['input'] + input['option'] + "\n Let's think step by step." 
        else:
            prompt = input['input'] + "\n Let's think step by step."
    elif task == 'PAS':
        if dataset == 'judgement':
            prompt = input['input'] + input['optimized_prompt'] + '\n' + input['option']
        else:
            prompt = input['input'] + '\n' + input['optimized_prompt']
    elif task == 'BPO' or task == 'FIPO':
        if dataset == 'dolly' or dataset =="self_instruct":
            prompt = input['optimized_prompt'] + '\n' + input['context']
        elif dataset == 'judgement':
            prompt = input['optimized_prompt'] + input['option']
        else:
            prompt = input['optimized_prompt']
    else:
        raise ValueError(f"Invalid task: {task}")

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.messages.create(
                model=model_name,
                max_tokens=512,
                temperature=0.9,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            end_time = time.time()
            
            output = response.content[0].text
            latency = round(end_time - start_time, 6)
            return prompt, output, latency
            
        except Exception as e:
            print(f"[Attempt {attempt + 1}/{max_retries}] Error in generate_supplementary: {repr(e)}")
            
            if attempt < max_retries - 1:  # 마지막 시도가 아닌 경우에만 대기
                # 백오프 전략: 지수적으로 증가하되 최대 대기 시간 제한
                wait_time = min(wait_base * (2 ** attempt), max_wait)
                # 약간의 랜덤성 추가하여 동시 요청 시 충돌 방지
                wait_time = wait_time + random.uniform(0, wait_time * 0.1)
                
                print(f"Retrying after {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries exceeded.")
                return prompt, None, None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)
    parser.add_argument('--model_id', type=str, help='Anthropic model name (e.g., claude-3-haiku-20240307, claude-3-sonnet-20240229, claude-3-opus-20240229)')
    parser.add_argument('--task', type=str)
    parser.add_argument('--dataset', type=str)
    args = parser.parse_args()

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

    # Anthropic 모델명 유효성 검사
    model_name = args.model_id
    print(f"Using Anthropic model: {model_name}")

    with open(args.output_path, 'w', encoding='utf-8') as outfile:
        for data in tqdm(dataset, desc='Generating Supplementary Prompt'):
            total += 1
            if args.task == 'LLPO':
                output, latency, p, sys_p = generate_supplementary_ours(data, model_name, dataset=args.dataset)
                if output is not None: # generate_supplementary_ours가 None을 반환하면 성공으로 간주하지 않음
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
                        'prompt': p,
                        'system_prompt': sys_p,
                        'output': 'Failed to generate output after multiple retries',
                        'latency': latency
                    }
            else:
                prompt, output, latency = generate_supplementary(data, model_name, args.task, dataset=args.dataset)
                if output is not None:  # generate_supplementary가 None을 반환하면 성공으로 간주하지 않음
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
                        'output': 'Failed to generate output after multiple retries',
                        'latency': latency
                    }
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print('\n--- Summary ---')
    print(f'Total: {total} | Success: {success} | Failed: {failed}')

if __name__ == '__main__':
    main()


# 사용 예시:
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/Ori/optimized_prompt/arena-hard_495.json --output_path evaluation/outputs/Ori/inferenced_outputs/arena_hard/Ori_Claude-3-Sonnet_outputs.jsonl --model_id claude-3-5-haiku-latest --task Ori
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/Ori/optimized_prompt/dolly.json --output_path evaluation/outputs/Ori/inferenced_outputs/dolly/Ori_Claude-3-Sonnet_outputs.jsonl --model_id claude-3-5-haiku-latest --task Ori
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/Ori/optimized_prompt/koala.json --output_path evaluation/outputs/Ori/inferenced_outputs/koala/Ori_Claude-3-Sonnet_outputs.jsonl --model_id claude-3-5-haiku-latest --task Ori
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/Ori/optimized_prompt/self_instruct.json --output_path evaluation/outputs/Ori/inferenced_outputs/self_instruct/Ori_Claude-3-Sonnet_outputs.jsonl --model_id claude-3-5-haiku-latest --task Ori
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/ModernBERT/ModernBERT_predictions_arena-hard.json --output_path evaluation/outputs/LLPO/kmeans/inferenced_outputs/ModernBERT/arena_hard/ModernBERT_LLPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task LLPO --dataset arena-hard
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/ModernBERT/ModernBERT_predictions_dolly.json --output_path evaluation/outputs/LLPO/kmeans/inferenced_outputs/ModernBERT/dolly/ModernBERT_LLPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task LLPO --dataset dolly
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/ModernBERT/ModernBERT_predictions_koala.json --output_path evaluation/outputs/LLPO/kmeans/inferenced_outputs/ModernBERT/koala/ModernBERT_LLPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task LLPO --dataset koala
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/ModernBERT/ModernBERT_predictions_self_instruct.json --output_path evaluation/outputs/LLPO/kmeans/inferenced_outputs/ModernBERT/self_instruct/ModernBERT_LLPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task LLPO --dataset self_instruct
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/BPO/optimized_prompt/arena-hard_495.json --output_path evaluation/outputs/BPO/inferenced_outputs/arena_hard/BPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task BPO
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/BPO/optimized_prompt/dolly.json --output_path evaluation/outputs/BPO/inferenced_outputs/dolly/context/BPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task BPO --dataset dolly
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/BPO/optimized_prompt/koala.json --output_path evaluation/outputs/BPO/inferenced_outputs/koala/BPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task BPO
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/BPO/optimized_prompt/self_instruct.json --output_path evaluation/outputs/BPO/inferenced_outputs/self_instruct/context/BPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task BPO --dataset self_instruct
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/CoT/optimized_prompt/arena-hard_495.json --output_path evaluation/outputs/CoT/inferenced_outputs/arena_hard/CoT_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task CoT
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/CoT/optimized_prompt/dolly.json --output_path evaluation/outputs/CoT/inferenced_outputs/dolly/CoT_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task CoT 
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/CoT/optimized_prompt/koala.json --output_path evaluation/outputs/CoT/inferenced_outputs/koala/CoT_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task CoT
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/CoT/optimized_prompt/self_instruct.json --output_path evaluation/outputs/CoT/inferenced_outputs/self_instruct/CoT_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task CoT 
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/PAS/optimized_prompt/arena-hard_495.json --output_path evaluation/outputs/PAS/inferenced_outputs/arena_hard/PAS_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task PAS
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/PAS/optimized_prompt/dolly.json --output_path evaluation/outputs/PAS/inferenced_outputs/dolly/PAS_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task PAS 
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/PAS/optimized_prompt/koala.json --output_path evaluation/outputs/PAS/inferenced_outputs/koala/PAS_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task PAS
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/PAS/optimized_prompt/self_instruct.json --output_path evaluation/outputs/PAS/inferenced_outputs/self_instruct/PAS_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task PAS 
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/FIPO/optimized_prompt/arena-hard_495.json --output_path evaluation/outputs/FIPO/inferenced_outputs/arena_hard/FIPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task FIPO
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/FIPO/optimized_prompt/dolly_context.json --output_path evaluation/outputs/FIPO/inferenced_outputs/dolly/context/FIPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task FIPO --dataset dolly
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/FIPO/optimized_prompt/koala.json --output_path evaluation/outputs/FIPO/inferenced_outputs/koala/FIPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task FIPO
# python evaluation/src/inference_Anthropic.py --input_path evaluation/outputs/FIPO/optimized_prompt/self_instruct_context.json --output_path evaluation/outputs/FIPO/inferenced_outputs/self_instruct/context/FIPO_Claude-3-5-Haiku_outputs.jsonl --model_id claude-3-5-haiku-latest --task FIPO --dataset self_instruct


