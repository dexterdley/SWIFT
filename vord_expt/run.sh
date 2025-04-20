################## SWIFT ##################
SIZE="3b"
################## SWIFT ##################

#--model deepseek-ai/deepseek-vl-7b-chat \
#--model swift/llava-v1.6-vicuna-7b-hf \
#--model AI-ModelScope/llava-onevision-qwen2-0.5b-ov-hf \
MODEL="deepseek-ai/deepseek-vl-7b-chat"

for PSI in 1
do
    MODEL_NAME="AI-ModelScope/${MODEL}_${PSI}-debug"
    MODEL_DIR="./checkpoints/$MODEL_NAME"

    echo "Training ${MODEL_NAME}"

    CUDA_VISIBLE_DEVICES=1\
    swift sft_vord \
        --model $MODEL \
        --dataset AI-ModelScope/LLaVA-Instruct-150K \
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
        --gradient_checkpointing_kwargs '{"use_reentrant": false}'
done
