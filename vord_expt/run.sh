################## SWIFT ##################
SIZE="7b"
MODEL_NAME="deepseek-vl-${SIZE}-finetune-vord2-max"
MODEL_DIR="./checkpoints/$MODEL_NAME"
################## SWIFT ##################

#--model deepseek-ai/deepseek-vl-7b-chat \
#--model swift/llava-v1.6-vicuna-7b-hf \

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
swift sft_vord \
    --model deepseek-ai/deepseek-vl-7b-chat \
    --dataset AI-ModelScope/LLaVA-Instruct-150K \
    --torch_dtype bfloat16 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 1 \
    --gradient_checkpointing True \
    --output_dir $MODEL_DIR \
    --num_train_epochs 1 \
    --save_steps 10000 \
    --power 2 \
    --sim_margin True \
    --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    # --deepspeed zero2
