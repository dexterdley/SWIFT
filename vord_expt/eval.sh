BACKBONE="deepseek-ai/deepseek-vl-7b-chat"

for MODEL in deepseek-vl-7b-chat-finetune-newvord0-margin-diffusion/v0-20250331-210211 deepseek-vl-7b-chat-finetune-newvord1-margin-diffusion/v0-20250330-220307 deepseek-vl-7b-chat-finetune-newvord2-margin-diffusion/v0-20250331-072140
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

    for DATASET in MME POPE BLINK COCO_VAL RealWorldQA MMMU_TEST OCRBench
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET}"

        CUDA_VISIBLE_DEVICES=4 \
        swift eval \
            --model $BACKBONE \
            --eval_dataset $DATASET \
            --eval_backend VLMEvalKit \
            --ckpt_dir $MODEL_DIR
    done
done
