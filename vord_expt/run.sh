################## SWIFT ##################
SIZE="3b"
################## SWIFT ##################

#--model deepseek-ai/deepseek-vl-7b-chat \
#--model swift/llava-v1.6-vicuna-7b-hf \
#--model AI-ModelScope/llava-onevision-qwen2-0.5b-ov-hf \
MODEL="AI-ModelScope/paligemma-3b-pt-224"

for PSI in 0
do
    MODEL_NAME="AI-ModelScope/${MODEL}_${PSI}-debug"
    MODEL_DIR="./checkpoints/$MODEL_NAME"

    echo "Training ${MODEL_NAME}"

    CUDA_VISIBLE_DEVICES=0\
    swift sft_vord \
        --model $MODEL \
        --dataset AI-ModelScope/LLaVA-Instruct-150K \
        --train_type full \
        --learning_rate 1e-5 \
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
        --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
        --eval_limit 100 \
        --eval_datasets realWorldQA \
        # --deepspeed zero2
done
