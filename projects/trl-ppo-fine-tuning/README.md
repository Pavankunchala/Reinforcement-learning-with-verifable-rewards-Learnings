# TRL PPO Fine-Tuning Example

This project demonstrates how to fine-tune a language model using Proximal Policy Optimization (PPO) with the `trl` library. The script is designed to work with models and datasets from the Hugging Face Hub or from local storage.

## Setup

1.  It is highly recommended to use a Python virtual environment.
2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

The `train.py` script accepts several command-line arguments to specify the model, dataset, and training parameters. Below are some common usage examples.

### Example 1: Fine-Tuning with a Hugging Face Model and Dataset

This command downloads a model and a dataset from the Hugging Face Hub, then starts the fine-tuning process.

```bash
python train.py \
    --base-model "meta-llama/Llama-2-7b-chat-hf" \
    --dataset "lvwerra/stack-exchange-paired" \
    --output-dir "./checkpoints/llama2-stack-exchange" \
    --use-4bit \
    --lora-r 16 \
    --lr 5e-6 \
    --hub-token "YOUR_HF_TOKEN_HERE"
```

### Example 2: Fine-Tuning with a Local Model and Local JSONL Data

This command uses a model that you have already downloaded to your local machine and a local `.jsonl` file for training data. The `--local-only` flag ensures no external connections are made.

```bash
python train.py \
    --base-model "/path/to/your/local/model" \
    --train-jsonl "/path/to/your/training_data.jsonl" \
    --eval-jsonl "/path/to/your/validation_data.jsonl" \
    --output-dir "./checkpoints/local-model-local-data" \
    --local-only \
    --use-4bit
```

## Key Arguments

#### Model & Data Arguments
*   `--base-model` (Required): The identifier for the model on the Hugging Face Hub (e.g., `codellama/CodeLlama-7b-hf`) or the absolute path to a local model directory.
*   `--dataset`: The identifier for a dataset on the Hugging Face Hub. Use this OR the local data arguments below.
*   `--train-jsonl`: The path to a local training data file in JSON Lines format.
*   `--eval-jsonl`: The path to a local validation data file in JSON Lines format.
*   `--output-dir` (Required): The directory where the trained model checkpoints will be saved.

#### System & Performance Arguments
*   `--local-only`: If set, the script will only use local files and not attempt to download models or datasets.
*   `--use-4bit`: If set, enables 4-bit quantization to reduce memory usage.
*   `--bf16` / `--fp16`: Use bfloat16 or float16 precision. Defaults to bf16 if available.
*   `--attn-impl`: The attention implementation to use (e.g., `sdpa` for scaled dot product attention).

#### Training & LoRA Arguments
*   `--lora-off`: If set, disables LoRA and performs full fine-tuning.
*   `--lora-r`: The rank of the LoRA matrices. Default is `16`.
*   `--lora-alpha`: The alpha parameter for LoRA scaling. Default is `32.0`.
*   `--lr`: The learning rate. Default is `5e-6`.
*   `--num-epochs`: The number of training epochs. Default is `1`.