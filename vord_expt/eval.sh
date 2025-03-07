################## SWIFT ##################
SIZE="7b"
MODEL_NAME="deepseek-vl-${SIZE}-finetune"
MODEL_DIR="./checkpoints/deepseek-vl-7b-finetune-vord1/v0-20250228-162922/checkpoint-28986/"
################## SWIFT ##################

# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \

for MODEL_DIR in ./checkpoints/deepseek-vl-7b-finetune-vord1-max/v0-20250306-115606/checkpoint-9662/ ./checkpoints/deepseek-vl-7b-finetune-vord2-max/v1-20250306-210746/checkpoint-9662/
do
    echo $MODEL_DIR

    for DATASET in BLINK
    do
        CUDA_VISIBLE_DEVICES=1 \
        swift eval \
            --model deepseek-ai/deepseek-vl-7b-chat \
            --eval_dataset $DATASET \
            --eval_backend VLMEvalKit \
            --ckpt_dir $MODEL_DIR
    done
done
