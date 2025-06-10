#!/bin/bash
################## SWIFT ##################
# SIZE="3b"
# --model deepseek-ai/deepseek-vl-7b-chat \
# --model swift/llava-v1.6-vicuna-7b-hf \
# MODEL="AI-ModelScope/paligemma-3b-pt-224"
################## SWIFT ##################

MODELS=(
  "AI-ModelScope/paligemma-3b-pt-224"
  "AI-ModelScope/paligemma2-3b-pt-224"
  # "deepseek-ai/deepseek-vl-7b-chat"
)
DATASET="AI-ModelScope/LLaVA-Instruct-150K"

ALGORITHMS=("BASE" "VCD" "VORD")
SEEDS=(42) # Can add 55 69 back if needed
NOISE=500

for MODEL in "${MODELS[@]}"; do
  for ALGO in "${ALGORITHMS[@]}"; do 
    for SEED in "${SEEDS[@]}"; do
      if [[ "$MODEL" == *"paligemma"* ]]; then
        PSI_VALUES=(0)
      elif [[ "$MODEL" == *"deepseek"* ]]; then
        PSI_VALUES=(0)
      else
        PSI_VALUES=(0) # Default PSI values if the model doesn't match
      fi

      for PSI in "${PSI_VALUES[@]}"; do
        # Extract the model name for the output directory
        MODEL_BASENAME=$(basename "$MODEL")
        MODEL_NAME="${DATASET}/${MODEL_BASENAME}-finetune-vord${PSI}-margin-diffusion-mask-decode-${ALGO}-${NOISE}"
        MODEL_DIR="./checkpoints/$MODEL_NAME"
        LOGGING_DIR="./runs/$MODEL_NAME"

        echo "Training: ${MODEL_NAME}, ${DATASET} with PSI=${PSI}, ALGO=${ALGO}, SEED=${SEED}"

        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
        NPROC_PER_NODE=8 \
        swift sft_vord \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --torch_dtype bfloat16 \
            --per_device_train_batch_size 4 \
            --per_device_eval_batch_size 4 \
            --gradient_checkpointing True \
            --output_dir "$MODEL_DIR" \
            --num_train_epochs 3 \
            --eval_steps 1000 \
            --save_steps 5000 \
            --power $PSI \
            --seed $SEED \
            --sim_margin True \
            --logging_dir "$LOGGING_DIR" \
            --eval_limit 100 \
            --eval_datasets MMStar \
            --deepspeed zero2 \
            --add_version False \
            --algo $ALGO \
            --noise $NOISE \
            --report_to "tensorboard" "wandb"

        CKPT_DIR="${MODEL_DIR}/checkpoint-38649/"
        if [ -d "$CKPT_DIR" ]; then
          for TESTSET in MME POPE BLINK HallusionBench MMStar; do # You can add "RealWorldQA" back if needed
            echo "EVALUATING: ${CKPT_DIR}, ${TESTSET}"
            
            CUDA_VISIBLE_DEVICES=7 \
            swift eval \
                  --model "$MODEL" \
                  --eval_dataset "$TESTSET" \
                  --eval_backend "VLMEvalKit" \
                  --ckpt_dir "$CKPT_DIR" \
                  --max_new_tokens 10
          done
        else
          echo "Checkpoint directory not found: $CKPT_DIR"
        fi
      done
    done
  done
done

# Results collection
for MODEL in "${MODELS[@]}"; do 
  for ALGO in "${ALGORITHMS[@]}"; do
    MODEL_BASENAME=$(basename "$MODEL")
    MODEL_NAME="${DATASET}/${MODEL_BASENAME}-finetune-vord0-margin-diffusion-mask-decode-vord-${ALGO}-${SEED}"
    MODEL_DIR="./checkpoints/$MODEL_NAME"
    RESULT_FILE="${MODEL_DIR}/checkpoint-19324/eval_result.jsonl"
    
    if [ -f "$RESULT_FILE" ]; then
      echo "Results for ${MODEL_NAME}:"
      cat "$RESULT_FILE"
      echo ""
    else
      echo "Results not found for ${MODEL_NAME}"
    fi
  done
done