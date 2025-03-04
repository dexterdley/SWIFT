################## SWIFT ##################
PROMPT_VERSION="v1"
SIZE="7b"
MODEL_NAME="deepseek-vl-${SIZE}-finetune"
MODEL_DIR="./checkpoints/deepseek-vl-7b-finetune-vord1/v0-20250228-162922/checkpoint-28986/"
################## SWIFT ##################

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
# --ckpt_dir checkpoints/deepseek-vl-7b-finetune/v0-20250221-215854/checkpoint-28986/\

for MODEL_DIR in ./checkpoints/deepseek-vl-7b-finetune-base/v0-20250303-211235/checkpoint-9662/ ./checkpoints/deepseek-vl-7b-finetune-vord1-max/v0-20250304-034307/checkpoint-9662
do
    echo $MODEL_DIR

    for DATASET in MME #POPE
    do
        CUDA_VISIBLE_DEVICES=0 \
        swift eval \
            --model deepseek-ai/deepseek-vl-7b-chat \
            --eval_dataset $DATASET \
            --ckpt_dir $MODEL_DIR
    done
done