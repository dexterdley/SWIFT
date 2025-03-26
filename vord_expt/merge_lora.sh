# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
MODEL="paligemma-3b-pt-224-finetune-vord1-margin-mix-diffusion/v2-20250326-122428"
MODEL_DIR="./checkpoints/AI-ModelScope/LLaVA-Instruct-150K/${MODEL}/checkpoint-9662/"

swift export \
    --adapters $MODEL_DIR \
    --merge_lora true
echo "MERGED"