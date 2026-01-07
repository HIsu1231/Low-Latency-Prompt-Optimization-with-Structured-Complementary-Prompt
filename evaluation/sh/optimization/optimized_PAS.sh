python evaluation/optimization/PAS_optimization.py \
  --dataset dolly \
  --input_path evaluation/eval_dataset/dolly.json \
  --output_path evaluation/outputs/PAS/optimized_prompt/dolly.json \

python evaluation/optimization/PAS_optimization.py \
  --dataset koala \
  --input_path evaluation/eval_dataset/koala.json \
  --output_path evaluation/outputs/PAS/optimized_prompt/koala.json 

python evaluation/optimization/PAS_optimization.py \
  --dataset arena-hard \
  --input_path evaluation/eval_dataset/arena-hard_495.json \
  --output_path evaluation/outputs/PAS/optimized_prompt/arena-hard_495.json \


python evaluation/optimization/PAS_optimization.py \
  --dataset self_instruct \
  --input_path evaluation/eval_dataset/self_instruct.json \
  --output_path evaluation/outputs/PAS/optimized_prompt/self_instruct.json\