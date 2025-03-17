################## SWIFT ##################
SIZE="7b"
################## SWIFT ##################

#--model deepseek-ai/deepseek-vl-7b-chat \
#--model swift/llava-v1.6-vicuna-7b-hf \
#--model AI-ModelScope/llava-onevision-qwen2-0.5b-ov-hf \

for PSI in 0 2
do
    MODEL_NAME="AI-ModelScope/llava-onevision-qwen2-0.5b-ov-hf${PSI}-debug"
    MODEL_DIR="./checkpoints/$MODEL_NAME"

    echo "Training ${MODEL_NAME}"

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        NPROC_PER_NODE=8 \
        swift sft_vord \
            --model deepseek-ai/deepseek-vl-7b-chat \
            --dataset swift/llava-instruct-mix-vsft \
            --torch_dtype bfloat16 \
            --per_device_train_batch_size 8 \
            --per_device_eval_batch_size 1 \
            --gradient_checkpointing True \
            --output_dir $MODEL_DIR \
            --num_train_epochs 1 \
            --learning_rate 5e-5 \
            --save_steps 1000 \
            --power $PSI \
            --sim_margin True \
            --logging_dir ./runs/$MODEL_NAME \
            --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
            # --deepspeed zero2
done
