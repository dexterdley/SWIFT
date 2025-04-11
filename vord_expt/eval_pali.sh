################## SWIFT ##################
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
################## SWIFT ##################
for MODEL in paligemma-3b-pt-224-finetune-vord1-max-margin-diffusion-acc-mask/v0-20250411-134017/
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

    for DATASET in MME #RealWorldQA
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

        CUDA_VISIBLE_DEVICES=6 \
        swift eval \
            --model AI-ModelScope/paligemma-3b-pt-224 \
            --eval_dataset "$DATASET" \
            --eval_backend VLMEvalKit \
            --ckpt_dir "$MODEL_DIR" \
            --max_new_tokens 10
    done
done