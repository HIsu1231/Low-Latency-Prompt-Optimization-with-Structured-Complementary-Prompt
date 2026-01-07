#Qwen
##self_instruct
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/self_instruct/PAS_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task PAS \
  --dataset self_instruct

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/self_instruct/PAS_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task PAS \
  --dataset self_instruct

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/arena_hard/PAS_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task PAS \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/arena_hard/PAS_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task PAS 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/arena_hard/PAS_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task PAS 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/arena_hard/PAS_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task PAS 

## dolly
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/dolly/PAS_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task PAS \
  --dataset dolly

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/dolly/PAS_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task PAS \
  --dataset dolly

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/dolly/PAS_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task PAS \
  --dataset dolly

##koala
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/koala.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/koala/PAS_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task PAS 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/koala.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/koala/PAS_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task PAS 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/koala.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/koala/PAS_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task PAS 



#OpenAI
##self_instruct
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/RoBERTa/RoBERTa_predictions_self_instruct.json\
  --output_path evaluation/outputs/PAS/inferenced_outputs/self_instruct/PAS_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task PAS \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/arena_hard/PAS_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task PAS 

## dolly
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/dolly/PAS_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task PAS \
  --dataset dolly

##koala
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/koala.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/koala/PAS_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task PAS \
  --dataset koala


#Claude
##self_instruct
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/self_instruct.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/self_instruct/PAS_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task PAS \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/arena-hard_495.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/arena_hard/PAS_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task PAS 

## dolly
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/dolly.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/dolly/PAS_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task PAS \
  --dataset dolly

##koala
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/PAS/optimized_prompt/koala.json \
  --output_path evaluation/outputs/PAS/inferenced_outputs/koala/PAS_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task PAS \
  --dataset koala
