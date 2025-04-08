################## SWIFT ##################
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
# --model AI-ModelScope/paligemma-3b-pt-224 \
################## SWIFT ##################

for MODEL in paligemma-3b-pt-224-finetune-newvord1-margin-diffusion-acc-mask/v0-20250407-215910/ paligemma-3b-pt-224-finetune-newvord2-margin-diffusion-acc-mask/v0-20250407-235531/
#for MODEL in deepseek-vl-7b-chat-finetune-newvord1-margin-diffusion/v1-20250404-221248 deepseek-vl-7b-chat-finetune-newvord2-margin-diffusion-correct-vit/v0-20250403-050948; 
do
  
    if [[ "$MODEL" =~ paligemma ]]; then
        BACKBONE="AI-ModelScope/paligemma-3b-pt-224"
    elif [[ "$MODEL" =~ deepseek-vl-7b ]]; then
        BACKBONE="deepseek-ai/deepseek-vl-7b-chat"
    else
        echo "WARNING: Model $MODEL does not match any backbone Skipping."
        continue # Skip to the next iteration of the loop
    fi

    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

    for DATASET in COCO_VAL #POPE BLINK HallusionBench; #MME RealWorldQA
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

        CUDA_VISIBLE_DEVICES=1 \
        swift eval \
            --model "$BACKBONE" \
            --eval_dataset "$DATASET" \
            --eval_backend VLMEvalKit \
            --ckpt_dir "$MODEL_DIR"

    done
done