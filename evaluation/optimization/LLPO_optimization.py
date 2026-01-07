#!/usr/bin/env python3
import os
os.environ['HF_HOME'] = '/nas/user77/workspace/models'
os.environ['HF_CACHE'] = '/nas/user77/workspace/models'
os.environ['TRANSFORMERS_CACHE'] = '/nas/user77/workspace/models'
import json
import time
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from train_with_accumulation_a5000 import MultiTaskBERT


def invert_mapping(label2idx):
    # JSON keys are strings, convert back to int
    return {
        task: {int(idx): lbl for lbl, idx in lbl_map.items()}
        for task, lbl_map in label2idx.items()
    }


def main(args):

    if args.data_path.endswith('.jsonl'):
        data = []
        with open(args.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data.append(json.loads(line))
    else:
        with open(args.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

    if args.task == "judgement":
            data = data.get("examples", [])

    if args.task in ['dolly', 'self_instruct']:
        texts = [(item['instruction'] + '\n' + item['context']).strip() for item in data]
    elif args.task == 'arena-hard':
        texts = [item['conversation'][0]['content'] for item in data]
    elif args.task == 'complex':
        texts = [item['instruction_en'] for item in data]
    elif args.task == 'judgement':
        texts = [item['input'] for item in data]
    elif args.task == 'gsm8k':
        texts = [item['question'] for item in data]
    else:
        texts = [(item['instruction']).strip() for item in data]

    label2idx_path = os.path.join(args.model_dir, 'label2idx.json')
    num_classes_path = os.path.join(args.model_dir, 'num_classes.json')

    # 2) Load mappings
    with open(label2idx_path, 'r', encoding='utf-8') as f:
        label2idx = json.load(f)
    with open(num_classes_path, 'r', encoding='utf-8') as f:
        num_classes = json.load(f)
    idx2label = invert_mapping(label2idx)

    # 3) Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = MultiTaskBERT(
        args.model_name,
        num_classes,
        dropout_rate=args.dropout_rate,
        gamma=args.gamma
    ).to(device)

    model_ckpt = os.path.join(args.model_dir, 'best_model.pt')
    model.load_state_dict(torch.load(model_ckpt, map_location=device))
    model.eval()

    # 4) Warm-up (오버헤드 제거용, 결과는 무시)
    print("Warming up model to eliminate first-run overhead...")
    warmup_encoding = tokenizer(
        texts[0],
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        _ = model(input_ids=warmup_encoding['input_ids'], attention_mask=warmup_encoding['attention_mask'])

    # 5) True Inference with latency measurement
    results = []
    with torch.no_grad():
        for idx, text in enumerate(tqdm(texts, desc="Inference"), start=1):
            # 토크나이즈 및 GPU 이동
            encoding = tokenizer(
                text,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt"
            ).to(device)

            # --- latency 측정 시작 ---
            start_time = time.time()
            outputs = model(input_ids=encoding['input_ids'], attention_mask=encoding['attention_mask'])
            end_time = time.time()
            latency = round(end_time - start_time, 6)
            # -------------------------

            # 예측 결과 디코딩
            pred = {
                task: idx2label[task][torch.argmax(logits[0]).item()]
                for task, logits in outputs.items() if task != 'loss'
            }

            results.append({
                'idx': idx,
                'input': text,
                'prediction': pred,
                'latency': latency
            })

    # 6) Save to file
    # output_path = os.path.join(args.output_path, args.task)
    # os.makedirs(output_path, exist_ok=True)
    if args.model_name == 'answerdotai/ModernBERT-large':
        output_file = os.path.join(args.output_path, f'ModernBERT_predictions_{args.task}.json')
    elif args.model_name == 'microsoft/deberta-v3-large':
        output_file = os.path.join(args.output_path, f'DeBERTa_predictions_{args.task}.json')
    else:
        output_file = os.path.join(args.output_path, f'RoBERTa_predictions_{args.task}.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved inference results to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',       type=str, default='/home/user77/2025EMNLP/new_evaluation/original_dataset/arena-hard/arena-hard_eval.json',
                        help='Path to JSON file with inputs (list of {\"input\":...})')
    parser.add_argument('--model_dir',       type=str, default='/home/user77/2025EMNLP/before_clustering_roberta_model/roberta-large_bs16_lr3e-05_drop0.3_ep10_seed42_gm2.0',
                        help='Directory containing best_model.pt, num_classes.json, label2idx.json')
    parser.add_argument('--output_path',     type=str, default='/home/user77/2025EMNLP/before_clustering_roberta_model')
    parser.add_argument('--model_name',      type=str, default='roberta-large')
    parser.add_argument('--max_length',      type=int, default=512)
    parser.add_argument('--batch_size',      type=int, default=32)  # 유지 (사용 안함)
    parser.add_argument('--dropout_rate',    type=float, default=0.1)
    parser.add_argument('--gamma',           type=float, default=2.0)
    parser.add_argument('--task',            type=str, default='arena-hard')
    args = parser.parse_args()

    main(args)
