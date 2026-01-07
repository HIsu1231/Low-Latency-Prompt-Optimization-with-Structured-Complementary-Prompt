#Qwen
##self_instruct
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_self_instruct.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/self_instruct/LLPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task LLPO \
  --dataset self_instruct

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_self_instruct.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/self_instruct/LLPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task LLPO \
  --dataset self_instruct

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_self_instruct.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/arena_hard/LLPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task LLPO \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa/RoBERTa_predictions_arena-hard.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/arena_hard/LLPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task LLPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa/RoBERTa_predictions_arena-hard.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/arena_hard/LLPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task LLPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa/RoBERTa_predictions_arena-hard.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/arena_hard/LLPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task LLPO 

## dolly
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa/RoBERTa_predictions_dolly.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/dolly/LLPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task LLPO \
  --dataset dolly

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_dolly.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/dolly/LLPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task LLPO \
  --dataset dolly

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_dolly.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/dolly/LLPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task LLPO \
  --dataset dolly

##koala
python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa/RoBERTa_predictions_koala.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/koala/LLPO_Qwen-2-0.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-0.5B-Instruct \
  --task LLPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa/RoBERTa_predictions_koala.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/koala/LLPO_Qwen-2-1.5B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-1.5B-Instruct \
  --task LLPO 

python evaluation/inference/inference_opensource.py \
  --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa/RoBERTa_predictions_koala.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/koala/LLPO_Qwen-2-7B-Instruct_outputs.jsonl \
  --model_id Qwen/Qwen2-7B-Instruct \
  --task LLPO 





#OpenAI
##self_instruct
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_self_instruct.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/self_instruct/LLPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task LLPO \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_arena-hard.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/arena_hard/LLPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task LLPO 

## dolly
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_dolly.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/dolly/LLPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task LLPO \
  --dataset dolly

##koala
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa/RoBERTa_predictions_koala.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/koala/LLPO_GPT_3.5_turbo_outputs.jsonl \
  --model_id gpt-3.5-turbo \
  --task LLPO \
  --dataset koala


#Claude
##self_instruct
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_self_instruct.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/self_instruct/LLPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task LLPO \
  --dataset self_instruct

##arena-hard
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_arena-hard.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/arena_hard/LLPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task LLPO 

## dolly
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/LLPO/optimized_prompt/RoBERTa/RoBERTa_predictions_dolly.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/dolly/LLPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task LLPO \
  --dataset dolly

##koala
python evaluation/inference/inference_OpenAI.py \
  --input_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa/RoBERTa_predictions_koala.json\
  --output_path evaluation/outputs/LLPO/inferenced_outputs/koala/LLPO_Claude-3-5-Haiku_outputs.jsonl \
  --model_id claude-3-5-haiku-latest \
  --task LLPO \
  --dataset koala
