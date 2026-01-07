python evaluation/optimization/LLPO_optimization.py\
    --data_path evaluation/eval_dataset/self_instruct.json\
    --model_dir Classifier/after_clustering/bestmodel/kmeans/FacebookAI_roberta_large_bs8_acc8_lr0.0001_drop0.3_ep15_seed42_gm2.0\
    --output_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa\
    --model_name "FacebookAI/roberta-large"\
    --max_length 512\
    --batch_size 64\
    --dropout_rate 0.3\
    --gamma 2.0\
    --task "self_instruct"


python evaluation/optimization/LLPO_optimization.py\
    --data_path evaluation/eval_dataset/dolly.json\
    --model_dir Classifier/after_clustering/bestmodel/kmeans/FacebookAI_roberta_large_bs8_acc8_lr0.0001_drop0.3_ep15_seed42_gm2.0\
    --output_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa\
    --model_name "FacebookAI/roberta-large"\
    --max_length 512\
    --batch_size 64\
    --dropout_rate 0.3\
    --gamma 2.0\
    --task "dolly"

python evaluation/optimization/LLPO_optimization.py\
    --data_path evaluation/eval_dataset/koala.json\
    --model_dir Classifier/after_clustering/bestmodel/kmeans/FacebookAI_roberta_large_bs8_acc8_lr0.0001_drop0.3_ep15_seed42_gm2.0\
    --output_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa\
    --model_name "FacebookAI/roberta-large"\
    --max_length 512\
    --batch_size 64\
    --dropout_rate 0.3\
    --gamma 2.0\
    --task "koala"

python evaluation/optimization/LLPO_optimization.py\
    --data_path evaluation/eval_dataset/self_instruct.json\
    --model_dir Classifier/after_clustering/bestmodel/kmeans/FacebookAI_roberta_large_bs8_acc8_lr0.0001_drop0.3_ep15_seed42_gm2.0\
    --output_path evaluation/outputs/LLPO/kmeans/optimized_prompt/RoBERTa\
    --model_name "FacebookAI/roberta-large"\
    --max_length 512\
    --batch_size 64\
    --dropout_rate 0.3\
    --gamma 2.0\
    --task "self_instruct"