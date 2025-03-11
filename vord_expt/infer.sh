################## SWIFT ##################
SIZE="7b"
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
MODEL_NAME="deepseek-vl-${SIZE}-finetune-base"
MODEL_DIR="./checkpoints/$MODEL_NAME"
################## SWIFT ##################
# cp ~/SWIFT/checkpoints/deepseek-vl-7b-finetune-base/v0-20250303-211235/runs/base/ ./checkpoints/deepseek-vl-7b-finetune-vord1-max/v0-20250304-034307/runs/
# AI-ModelScope/LLaVA-Instruct-150K
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --adapters ./checkpoints/deepseek-vl-7b-finetune-vord2-max-mix/v1-20250310-064047/checkpoint-9662/ \
    --stream true \
    --temperature 1.0 \
    --max_new_tokens 512 \
    --return_dict_in_generate true \
    --output_scores true
