BACKBONE="AI-ModelScope/paligemma-3b-pt-224"

#for MODEL in deepseek-vl-7b-chat-finetune-newvord0-margin-diffusion/v0-20250331-210211 deepseek-vl-7b-chat-finetune-newvord1-margin-diffusion/v0-20250330-220307 deepseek-vl-7b-chat-finetune-newvord2-margin-diffusion/v0-20250331-072140
for MODEL in paligemma-3b-pt-224-finetune-newvord1-grad-margin-diffusion/v0-20250402-101536/ paligemma-3b-pt-224-finetune-newvord2-grad-margin-diffusion/v0-20250402-125404/
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

    for DATASET in MME #POPE BLINK COCO_VAL RealWorldQA MMMU_TEST OCRBench
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
