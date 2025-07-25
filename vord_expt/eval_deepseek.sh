################## SWIFT ##################
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
# --model AI-ModelScope/paligemma-3b-pt-224 \
################## SWIFT ##################

for MODEL in deepseek-vl-7b-chat-finetune-vord0-margin-diffusion-mask-decode-BASE-500-55 deepseek-vl-7b-chat-finetune-vord0-margin-diffusion-mask-decode-VORD-500-55
do
  
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-28986/"

    for DATASET in POPE BLINK HallusionBench MMStar
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

        CUDA_VISIBLE_DEVICES=1 \
        swift eval \
            --model deepseek-ai/deepseek-vl-7b-chat \
            --eval_dataset "$DATASET" \
            --eval_backend VLMEvalKit \
            --ckpt_dir "$MODEL_DIR" \
            --logprobs True
    done
done

'''
MODEL_DIR="./old_runs/checkpoints/deepseek-vl-7b-finetune-base/v0-20250303-211235/checkpoint-9662/"

for DATASET in BLINK HallusionBench;
do
    echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

    CUDA_VISIBLE_DEVICES=1 \
    swift eval \
        --model deepseek-ai/deepseek-vl-7b-chat \
        --eval_dataset "$DATASET" \
        --eval_backend VLMEvalKit \
        --ckpt_dir "$MODEL_DIR"
done

'''