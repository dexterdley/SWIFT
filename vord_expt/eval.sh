################## SWIFT ##################
MODEL_NAME="deepseek-vl-7b-full-finetune-vord2-margin-diffuse/v0-20250323-102849"
################## SWIFT ##################

# deepseek-vl-7b-finetune-vord0-max-mix/v0-20250311-215535
# deepseek-vl-7b-finetune-vord1-max-mix/v0-20250312-065804
# deepseek-vl-7b-finetune-vord2-max-mix/v0-20250312-162700

for MODEL in deepseek-vl-7b-full-finetune-vord0-margin/v0-20250320-044003 deepseek-vl-7b-full-finetune-vord2-margin-newmix/v0-20250320-215007 deepseek-vl-7b-full-finetune-vord1-margin-newmix/v1-20250321-065753 deepseek-vl-7b-full-finetune-vord1-margin-diffuse/v0-20250323-194649 deepseek-vl-7b-full-finetune-vord2-margin-diffuse/v0-20250323-102849
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

    for DATASET in MME POPE BLINK
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET}"

        CUDA_VISIBLE_DEVICES=4 \
        swift eval \
            --model deepseek-ai/deepseek-vl-7b-chat \
            --eval_dataset $DATASET \
            --eval_backend VLMEvalKit \
            --ckpt_dir $MODEL_DIR
    done
done
