################## SWIFT ##################
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
# --model AI-ModelScope/paligemma-3b-pt-224 \
################## SWIFT ##################

for MODEL in paligemma-3b-pt-224-finetune-vord0-margin-diffusion-mask-decode-VCD-500
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

    for DATASET in POPE BLINK HallusionBench MMStar #MME
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

        CUDA_VISIBLE_DEVICES=0 \
        swift eval \
            --model AI-ModelScope/paligemma-3b-pt-224 \
            --eval_dataset "$DATASET" \
            --eval_backend VLMEvalKit \
            --ckpt_dir "$MODEL_DIR" \
            --max_new_tokens 10
    done
done

for MODEL in paligemma2-3b-pt-224-finetune-vord0-margin-diffusion-mask-decode-VCD-500
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

    for DATASET in POPE BLINK HallusionBench MMStar
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

        CUDA_VISIBLE_DEVICES=1 \
        swift eval \
            --model AI-ModelScope/paligemma2-3b-pt-224 \
            --eval_dataset "$DATASET" \
            --eval_backend VLMEvalKit \
            --ckpt_dir "$MODEL_DIR" \
            --max_new_tokens 10
    done
done

for MODEL in deepseek-vl-7b-chat-finetune-vord0-margin-diffusion-mask-decode-VCD-500
do
    MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

    for DATASET in POPE BLINK HallusionBench MMStar
    do
        echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

        CUDA_VISIBLE_DEVICES=1 \
        swift eval \
            --model deepseek-ai/deepseek-vl-7b-chat \
            --eval_dataset "$DATASET" \
            --eval_backend VLMEvalKit \
            --ckpt_dir "$MODEL_DIR" \
            --max_new_tokens 10
    done
done