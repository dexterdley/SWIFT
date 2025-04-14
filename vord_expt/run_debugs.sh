MODELS=(
  "AI-ModelScope/paligemma-3b-pt-224"
  #"deepseek-ai/deepseek-vl-7b-chat"
  #deepseek-ai/deepseek-vl2-small
)
DATASET="AI-ModelScope/LLaVA-Instruct-150K"

for MODEL in "${MODELS[@]}"
do
  if [[ "$MODEL" == *"paligemma"* ]]; then
    PSI_VALUES=(0 1)
  elif [[ "$MODEL" == *"deepseek"* ]]; then
    PSI_VALUES=(0)
  else
    PSI_VALUES=(1) # Default PSI values if the model doesn't match
  fi

  for PSI in "${PSI_VALUES[@]}"
  do
      # Extract the model name for the output directory
      MODEL_BASENAME=$(basename "$MODEL")
      MODEL_NAME="${DATASET}/${MODEL_BASENAME}-finetune-newvord${PSI}-margin-diffusion-debug-mean"
      MODEL_DIR="./checkpoints/$MODEL_NAME"
      LOGGING_DIR="./runs/$MODEL_NAME"

      echo "Training: ${MODEL_NAME}, ${DATASET} with PSI=${PSI}"

      CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
      NPROC_PER_NODE=8 \
      swift sft_vord \
          --model "$MODEL" \
          --dataset "$DATASET" \
          --torch_dtype bfloat16 \
          --per_device_train_batch_size 8 \
          --per_device_eval_batch_size 8 \
          --gradient_checkpointing True \
          --output_dir "$MODEL_DIR" \
          --num_train_epochs 1 \
          --eval_steps 1000 \
          --save_steps 4000 \
          --max_length 1200 \
          --power $PSI \
          --sim_margin True \
          --logging_dir "$LOGGING_DIR" \
          --eval_limit 100 \
          --eval_datasets MMStar \
          --deepspeed zero1 \
          --max_steps 500 \
          --full_determinism True\
          --add_version False
      '''
      CKPT_DIR="${MODEL_DIR}/checkpoint-500/"
      for TESTSET in MME #RealWorldQA
      do
        echo "EVALUATING: ${CKPT_DIR}, ${DATASET} $BACKBONE"

        CUDA_VISIBLE_DEVICES=6 \
        swift eval \
            --model $MODEL \
            --eval_dataset "$TESTSET" \
            --eval_backend VLMEvalKit \
            --ckpt_dir "$CKPT_DIR" \
            --max_new_tokens 10
      done
      '''
  done
done