################## SWIFT ##################
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
################## SWIFT ##################
for MODEL in paligemma-3b-pt-224-finetune-newvord1-margin-diffusion-acc-mask/v0-20250407-215910/ paligemma-3b-pt-224-finetune-newvord2-margin-diffusion-acc-mask/v0-20250407-235531/
# paligemma-3b-pt-224-finetune-newvord2-grad-margin-diffusion/v0-20250402-125404 #paligemma-3b-pt-224-finetune-newvord0-margin-diffusion/v0-20250330-160209 paligemma-3b-pt-224-finetune-newvord1-margin-diffusion/v0-20250330-175349 paligemma-3b-pt-224-finetune-newvord2-margin-diffusion/v0-20250330-194809;
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

    for DATASET in MME #POPE BLINK HallusionBench MME RealWorldQA
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

        CUDA_VISIBLE_DEVICES=0 \
        swift eval \
            --model AI-ModelScope/paligemma-3b-pt-224 \
            --eval_dataset "$DATASET" \
            --eval_backend VLMEvalKit \
            --ckpt_dir "$MODEL_DIR"

    done
done