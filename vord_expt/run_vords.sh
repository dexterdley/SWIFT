################## SWIFT ##################
SIZE="7b"
MODEL="swift/llava-v1.6-vicuna-7b-hf"
DATASET="AI-ModelScope/LLaVA-Instruct-150K"
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
################## SWIFT ##################
# swift/ScienceQA
for PSI in 0 1 2
do
    MODEL_NAME="${DATASET}/llava-v1.6-${SIZE}-finetune-vord${PSI}-margin-mix-diffusion"
    MODEL_DIR="./checkpoints/$MODEL_NAME"

    echo "Training ${MODEL_NAME}, ${DATASET}"

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    NPROC_PER_NODE=8 \
    swift sft_vord \
        --model $MODEL \
        --dataset $DATASET\
        --torch_dtype bfloat16 \
        --per_device_train_batch_size 8 \
        --per_device_eval_batch_size 8 \
        --gradient_checkpointing True \
        --output_dir $MODEL_DIR \
        --num_train_epochs 1 \
        --save_steps 1000 \
        --power $PSI \
        --sim_margin True \
        --logging_dir ./runs/$MODEL_NAME \
        --deepspeed zero2
done
