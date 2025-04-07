################## SWIFT ##################
# SIZE="3b"
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
# MODEL="AI-ModelScope/paligemma-3b-pt-224"
################## SWIFT ##################

MODELS=(
  AI-ModelScope/paligemma2-3b-pt-224
  # AI-ModelScope/paligemma-3b-pt-224
  # deepseek-ai/deepseek-vl-7b-chat
  # Qwen/Qwen2.5-VL-3B-Instruct-AWQ
)
DATASET="AI-ModelScope/LLaVA-Instruct-150K"

for MODEL in "${MODELS[@]}"
do  
  for SEED in 42 #55 69
  do
    if [[ "$MODEL" == *"paligemma"* ]]; then
      PSI_VALUES=(0 1 2)
    elif [[ "$MODEL" == *"deepseek"* ]]; then
      PSI_VALUES=(0 1 2)
    else
      PSI_VALUES=(0 1 2) # Default PSI values if the model doesn't match
    fi

    for PSI in "${PSI_VALUES[@]}"
    do
        # Extract the model name for the output directory
        MODEL_BASENAME=$(basename "$MODEL")
        MODEL_NAME="${DATASET}/${MODEL_BASENAME}-finetune-vord${PSI}-margin-diffusion-mask"
        MODEL_DIR="./checkpoints/$MODEL_NAME"
        LOGGING_DIR="./runs/$MODEL_NAME"

        echo "Training: ${MODEL_NAME}, ${DATASET} with PSI=${PSI}"

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        NPROC_PER_NODE=8 \
        swift sft_vord \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --torch_dtype bfloat16 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 8 \
            --gradient_checkpointing True \
            --output_dir "$MODEL_DIR" \
            --num_train_epochs 1 \
            --eval_steps 1000 \
            --save_steps 5000 \
            --power $PSI \
            --sim_margin True \
            --logging_dir "$LOGGING_DIR" \
            --eval_limit 100 \
            --eval_datasets realWorldQA \
            --deepspeed zero1 \
            --data_seed $SEED \
            --report_to tensorboard wandb
    done
  done
done