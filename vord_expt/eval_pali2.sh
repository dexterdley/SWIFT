################## SWIFT ##################
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
################## SWIFT ##################

#for MODEL in paligemma-3b-pt-224-finetune-newvord1-margin-diffusion-mask/v0-20250406-222451 paligemma-3b-pt-224-finetune-newvord2-margin-diffusion-mask/v0-20250407-002546 # paligemma-3b-pt-224-finetune-newvord2-grad-margin-diffusion/v0-20250402-125404 #paligemma-3b-pt-224-finetune-newvord0-margin-diffusion/v0-20250330-160209 paligemma-3b-pt-224-finetune-newvord1-margin-diffusion/v0-20250330-175349 paligemma-3b-pt-224-finetune-newvord2-margin-diffusion/v0-20250330-194809;
for MODEL in paligemma2-3b-pt-224-finetune-vord0-margin-diffusion-mask/v0-20250407-071848/ paligemma2-3b-pt-224-finetune-vord1-margin-diffusion-mask/v0-20250407-110909/ paligemma2-3b-pt-224-finetune-vord2-margin-diffusion-mask/v0-20250407-145941/
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