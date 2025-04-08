################## SWIFT ##################
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
################## SWIFT ##################

for MODEL in paligemma2-3b-pt-224-finetune-vord1-margin-diffusion-mask/v1-20250408-034627
#for MODEL in paligemma2-3b-pt-224-finetune-vord0-margin-diffusion-mask/v0-20250407-071848/ paligemma2-3b-pt-224-finetune-vord1-margin-diffusion-mask/v0-20250407-110909/ paligemma2-3b-pt-224-finetune-vord2-margin-diffusion-mask/v0-20250407-145941/
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-19324/"

    for DATASET in MME #POPE BLINK HallusionBench MME RealWorldQA
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

        CUDA_VISIBLE_DEVICES=0 \
        swift eval \
            --model AI-ModelScope/paligemma2-3b-pt-224 \
            --eval_dataset "$DATASET" \
            --eval_backend VLMEvalKit \
            --ckpt_dir "$MODEL_DIR"

    done
done