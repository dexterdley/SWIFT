################## SWIFT ##################
SIZE="3b"
################## SWIFT ##################

#--model deepseek-ai/deepseek-vl-7b-chat \
#--model swift/llava-v1.6-vicuna-7b-hf \

MODEL="AI-ModelScope/paligemma-3b-pt-224"

DATASET="HuggingFaceH4/rlaif-v_formatted"  # This dataset has JSON parsing issues
# DATASET="AI-ModelScope/LLaVA-Instruct-150K"

for PSI in 1
do
    MODEL_NAME="AI-ModelScope/${MODEL}_${PSI}-debug-dpo"
    MODEL_DIR="./checkpoints/$MODEL_NAME"

    echo "Training ${MODEL_NAME}"

    CUDA_VISIBLE_DEVICES=1\
    swift rlhf\
        --rlhf_type dpo \
        --model $MODEL \
        --dataset "$DATASET" \
        --split_dataset_ratio 0.01 \
        --torch_dtype bfloat16 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_checkpointing True \
        --output_dir $MODEL_DIR \
        --num_train_epochs 1 \
        --save_steps 1000 \
        --logging_dir ./runs/$MODEL_NAME \
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --train_type lora \
        --max_steps 10 \
        --logging_steps 1 \
        --save_steps 5 \
        --freeze_vit true
done