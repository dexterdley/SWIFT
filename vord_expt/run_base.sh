################## SWIFT ##################
SIZE="7b"
MODEL_NAME="deepseek-vl-${SIZE}-finetune-base"
MODEL_DIR="./checkpoints/$MODEL_NAME"
DATASET="AI-ModelScope/paligemma-3b-pt-224"
################## SWIFT ##################

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
# --model Qwen/Qwen2.5-VL-3B-Instruct \

MODEL_NAME="${DATASET}/${MODEL_NAME}-finetune-newvord${PSI}-margin-diffusion-debug-mean-vord-BASE"
MODEL_DIR="./checkpoints/$MODEL_NAME"
LOGGING_DIR="./runs/$MODEL_NAME"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model AI-ModelScope/paligemma2-3b-pt-224 \
    --dataset AI-ModelScope/LLaVA-Instruct-150K \
    --train_type full \
    --learning_rate 1e-5 \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_checkpointing True \
    --num_train_epochs 1 \
    --max_steps 500 \
    --save_steps 20000 \
    --logging_dir "$LOGGING_DIR" \
    --output_dir $MODEL_DIR \
    --deepspeed zero2 \
    --add_version False
    # --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
