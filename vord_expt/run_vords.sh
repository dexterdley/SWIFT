################## SWIFT ##################
SIZE="7b"
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
MODEL_NAME="deepseek-vl-${SIZE}-finetune-base"
MODEL_DIR="./checkpoints/$MODEL_NAME"
################## SWIFT ##################
# cp ~/SWIFT/checkpoints/deepseek-vl-7b-finetune-base/v0-20250303-211235/runs/base/ ./checkpoints/deepseek-vl-7b-finetune-vord1-max/v0-20250304-034307/runs/

for PSI in 1 2
do
    MODEL_NAME="deepseek-vl-${SIZE}-finetune-vord${PSI}-max"
    MODEL_DIR="./checkpoints/$MODEL_NAME"

    echo "Training ${MODEL_NAME}"

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
            --save_steps 1000 \
            --power $PSI \
            --sim_margin True \
            --logging_dir ./runs/$MODEL_NAME \
            --deepspeed zero2
done
