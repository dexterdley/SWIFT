################## SWIFT ##################
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
# --model AI-ModelScope/paligemma-3b-pt-224 \

# HA-DPO: juliozhao/hadpo-llava-1.5
# OPA-DPO: zhyang2226/opadpo-lora_llava-v1.5-7b
# RLAIF: openbmb/RLAIF-V-7B
# YiyangAiLab/llava_POVID_stage_two_lora

# huggingface-cli download zhyang2226/opadpo-lora_llava-v1.5-7b --local-dir ./checkpoints
################## SWIFT ##################

MODEL_DIR="./checkpoints/HA_DPO"

for DATASET in POPE
do
    echo "EVALUATING: ${MODEL_DIR}, ${DATASET} $BACKBONE"

    CUDA_VISIBLE_DEVICES=3 \
    swift eval \
        --model llava-hf/llava-1.5-7b-hf \
        --eval_dataset "$DATASET" \
        --eval_backend VLMEvalKit \
        --ckpt_dir "$MODEL_DIR" \
        --merge_lora true \
        --max_new_tokens 10
done
