#Qwen
##self_instruct
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/self_instruct/FIPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task FIPO \
  --dataset self_instruct

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/self_instruct/FIPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task FIPO \
  --dataset self_instruct

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/arena_hard/FIPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task FIPO \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/arena_hard/FIPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task FIPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/arena_hard/FIPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task FIPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/arena_hard/FIPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task FIPO 

## dolly
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/dolly/FIPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task FIPO \
  --dataset dolly

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/dolly/FIPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task FIPO \
  --dataset dolly

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/dolly/FIPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task FIPO \
  --dataset dolly

##koala
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/koala/FIPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task FIPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/koala/FIPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task FIPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/koala/FIPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task FIPO 


#OpenAI
##self_instruct
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/self_instruct/FIPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task FIPO \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/arena_hard/FIPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task FIPO 

## dolly
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/dolly/FIPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task FIPO \
  --dataset dolly

##koala
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/koala/FIPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task FIPO \
  --dataset koala


#Claude
##self_instruct
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/self_instruct/FIPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task FIPO \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/arena_hard/FIPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task FIPO 

## dolly
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/dolly/FIPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task FIPO \
  --dataset dolly

##koala
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/FIPO/optimized_prompt/koala.json \
  --output_path evaluation/outputs/FIPO/inferenced_outputs/koala/FIPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task FIPO \
  --dataset koala
