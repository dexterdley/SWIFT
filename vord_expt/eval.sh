################## SWIFT ##################
SIZE="7b"
MODEL_NAME="deepseek-vl-${SIZE}-finetune"
MODEL_DIR="./checkpoints/deepseek-vl-7b-finetune-vord1/v0-20250228-162922/checkpoint-28986/"
################## SWIFT ##################

# ./checkpoints/deepseek-vl-7b-finetune-vord0-max-mix/v0-20250307-184642/checkpoint-9662/
# ./checkpoints/deepseek-vl-7b-finetune-vord1-max-mix/v0-20250308-035620/checkpoint-9662/
# ./checkpoints/deepseek-vl-7b-finetune-vord2-max-mix/v0-20250308-130307/checkpoint-9662/

for MODEL in deepseek-vl-7b-finetune-vord0-max-mix/v0-20250311-215535
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"
    echo $MODEL_DIR

    for DATASET in MME
    do
        CUDA_VISIBLE_DEVICES=1 \
        swift eval \
            --model deepseek-ai/deepseek-vl-7b-chat \
            --eval_dataset $DATASET \
            --eval_backend VLMEvalKit \
            --ckpt_dir $MODEL_DIR
    done
done
