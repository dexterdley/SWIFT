################## SWIFT ##################
SIZE="7b"
MODEL_NAME="deepseek-vl-${SIZE}-finetune-base"
MODEL_DIR="./checkpoints/$MODEL_NAME"
################## SWIFT ##################

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
# --model Qwen/Qwen2.5-VL-3B-Instruct \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft \
    --model deepseek-ai/deepseek-vl-7b-chat \
    --dataset AI-ModelScope/LLaVA-Instruct-150K \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_checkpointing True \
    --num_train_epochs 3 \
    --save_steps 20000 \
    --output_dir $MODEL_DIR \
    --deepspeed zero2
    # --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
