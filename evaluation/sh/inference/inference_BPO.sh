#Qwen
##self_instruct
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/self_instruct/BPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task BPO \
  --dataset self_instruct

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/self_instruct/BPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task BPO \
  --dataset self_instruct

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/arena_hard/BPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task BPO \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/arena_hard/BPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task BPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/arena_hard/BPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task BPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/arena_hard/BPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task BPO 

## dolly
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/dolly/BPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task BPO \
  --dataset dolly

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/dolly/BPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task BPO \
  --dataset dolly

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/dolly/BPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task BPO \
  --dataset dolly

##koala
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/koala/BPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task BPO \
  --dataset koala

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/koala/BPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task BPO \
  --dataset koala

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/koala/BPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task BPO \
  --dataset koala


#OpenAI
##self_instruct
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/self_instruct/BPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task BPO \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/arena_hard/BPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task BPO 

## dolly
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/dolly/BPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task BPO \
  --dataset dolly

##koala
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/koala/BPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task BPO \
  --dataset koala


#Claude
##self_instruct
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/self_instruct/BPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task BPO \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/arena_hard/BPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task BPO 

## dolly
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/dolly/BPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task BPO \
  --dataset dolly

##koala
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/BPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/BPO/inferenced_outputs/koala/BPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task BPO \
  --dataset koala
