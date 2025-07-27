# Mamba Poet: Fine-tuning Mamba for Poetry Generation

## Overview

This project fine-tunes a 130M parameter Mamba model (`state-spaces/mamba-130m-hf`) to generate poems based on user prompts (e.g., "Write a poem about topic X in Y style"). The process covers environment setup, dataset preparation (optionally synthetic generation), multi-GPU fine-tuning using Hugging Face `transformers` and `accelerate`, and instructions for running inference with the fine-tuned model.

This project was developed and tested with 3x NVIDIA RTX 3090 GPUs in a Linux environment.

## Setup

### Prerequisites
* Anaconda (or Miniconda)
* NVIDIA GPU(s) with appropriate CUDA drivers installed.

### 1. Create Conda Environment
Create and activate a new Conda environment (e.g., named `mamba_poet` with Python 3.9):
```bash
conda create -n mamba_poet python=3.9 -y
conda activate mamba_poet
````

### 2\. Install PyTorch with CUDA

Install PyTorch matching your system's CUDA version. Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) for the correct command. For example, for CUDA 12.1:

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### 3\. Install Other Dependencies

It's recommended to install dependencies from a `requirements.txt` file. Create a `requirements.txt` file with at least the following (or use `pip freeze > requirements.txt` in your working environment to capture all packages and their versions):

```txt
# requirements.txt (example content, refer to project for exact list)
transformers>=4.30.0 
accelerate>=0.20.0 
datasets
sentencepiece
mamba-ssm
causal-conv1d
einops
# For dataset generation (optional)
# google-generativeai 
# For monitoring (optional)
# tensorboard
```

Then install:

```bash
pip install -r requirements.txt
```

Alternatively, install key packages manually:

```bash
pip install transformers accelerate datasets sentencepiece mamba-ssm causal-conv1d einops
```

### 4\. (Optional) Set Environment Variable for Tokenizers

To avoid warnings from the `tokenizers` library during distributed training, you can set an environment variable. Add this at the beginning of your Python scripts or set it in your shell:

```python
# In your Python script (e.g., train_mamba_poet.py)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
```

Or in bash:

```bash
export TOKENIZERS_PARALLELISM=false
```

## Dataset

### Dataset Format

The model expects a dataset of prompt-poem pairs. This should be a `.jsonl` file where each line is a JSON object with two keys:

  * `"instruction_prompt"`: The text prompt given to the model (e.g., "Write a sonnet about the moon.").
  * `"poem"`: The corresponding poem text.

Example line in `your_dataset.jsonl`:

```json
{"instruction_prompt": "Compose a haiku about a blooming cherry tree.", "poem": "Pink petals flutter,\\nSpring's gentle breath whispers by,\\nBeauty takes its hold."}
```

(Note: `\\n` represents newline characters within the poem string in the JSONL file).

### Dataset Generation (Optional)

A script like `generate_poetry_data.py`  can be used to synthetically generate such a dataset using a teacher LLM (e.g., Gemini). This script would typically:

  * Take lists of topics and styles.
  * Prompt a teacher LLM to generate poems and corresponding instruction prompts.
  * Append these pairs to the specified `.jsonl` output file (e.g., `poetry_dataset_batch.jsonl`).
  * **Important:** Manually review the quality and diversity of any synthetically generated data.
  * A `sample_dataset.jsonl` is included here for reference. `poetry_dataset_batch.jsonl` is not included becuse of size

## Fine-tuning the Model

Fine-tuning is performed using the `train_mamba_poet.py` script.

### 1\. Prepare your Dataset File

Ensure your `.jsonl` dataset file (e.g., `poetry_dataset_batch.jsonl`) is ready and its path is correctly specified in `train_mamba_poet.py`.

### 2\. Configure `train_mamba_poet.py` (if needed)

Before running, you might want to review and modify settings within `train_mamba_poet.py`:

  * **`dataset_file`**: Path to your dataset.
  * **`model_checkpoint_name`**: Default is `state-spaces/mamba-130m-hf`.
  * **`output_directory`**: Where model checkpoints and logs will be saved (e.g., `./mamba_poet_finetuned_large_dataset_kst`). It's good practice to use a new directory for each training run.
  * **`TrainingArguments`**: Adjust hyperparameters like `num_train_epochs`, `per_device_train_batch_size`, `learning_rate`, etc.
      * It is recommended to keep `load_best_model_at_end=False` to avoid potential NCCL timeout issues observed with this setup. The best checkpoint can be selected manually based on logs.
      * Consider setting `ddp_find_unused_parameters=False` if you are sure your model has no unused parameters during the forward pass.

### 3\. Configure `accelerate` (One-time setup per machine)

If this is your first time using `accelerate` on this machine or for this project setup:

```bash
accelerate config
```

Follow the prompts. For a single machine with multiple GPUs (e.g., 3x RTX 3090):

  * Compute environment: `This machine`
  * Machine type: `multi-GPU`
  * Number of different machines: `1`
  * DeepSpeed/FSDP/Megatron-LM: `NO` (unless specifically configured)
  * Number of GPUs: `3` (or your actual count)
  * Mixed precision: `fp16` (suitable for RTX 3090s)
  * Answer NO to the rest or just press Enter for the default option (which will be in capital letters)
### 4\. Launch Training

Navigate to your project directory in the terminal and run:

```bash
accelerate launch --main_process_port 0 train_mamba_poet.py
```

  * `--main_process_port 0` helps automatically find a free port for distributed communication, resolving potential `ConnectionError`s.

### 5\. Monitoring

  * Training progress (loss, evaluation metrics) will be printed to the console.
  * If `report_to="tensorboard"` is set in `TrainingArguments`, view logs with:
    ```bash
    tensorboard --logdir <your_output_directory>/logs
    ```

### 6\. Output and Selecting the Best Model

  * Checkpoints are saved periodically in subdirectories like `<output_directory>/checkpoint-X`.
  * With `load_best_model_at_end=False`, the script saves the model state at the very end of training to `<output_directory>/final_best_model` (this represents the model at the last epoch).
  * **To get the actual best model:** Review the `eval_loss` reported for each epoch during training. Identify the checkpoint directory (e.g., `checkpoint-X`) that corresponds to the epoch with the lowest `eval_loss`. This is the checkpoint you should use for inference.

## Checking the Fine-tuned Model / Inference

Use a script (e.g., `check_model.py`, adapted from `chech_head.py`) to load a saved checkpoint and generate poems.

### 1\. `check_model.py` Script Example:

```python
# check_model.py
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# --- CONFIGURATION ---
# Path to your chosen fine-tuned model checkpoint directory
# (e.g., the best one identified from logs like "./mamba_poet_finetuned_large_dataset_kst/checkpoint-XYZ",
# or the one saved at the end: "./mamba_poet_finetuned_large_dataset_kst/final_best_model")
MODEL_PATH = "./mamba_poet_finetuned_large_dataset_kst/checkpoint-30" # MODIFY AS NEEDED
YOUR_PROMPT = "Write a thoughtful poem about the passage of time" # MODIFY AS NEEDED
# --- END CONFIGURATION ---

def main():
    # Optional: Set TOKENIZERS_PARALLELISM for cleaner logs if this script forks or uses tokenizers in parallel
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"

    absolute_model_path = os.path.abspath(MODEL_PATH)
    print(f"Attempting to load model from: {absolute_model_path}")

    if not os.path.isdir(absolute_model_path):
        print(f"Error: Directory not found at {absolute_model_path}")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(absolute_model_path)
        model = AutoModelForCausalLM.from_pretrained(absolute_model_path)
        print("Model and tokenizer loaded successfully.")

        # Verify lm_head (optional, for debugging)
        if 'lm_head.weight' in model.state_dict():
            print("🎉 'lm_head.weight' IS PRESENT.")
        else:
            print("😭 'lm_head.weight' IS MISSING. The model may not generate text correctly.")

        # Generation
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval() # Set model to evaluation mode

        inputs = tokenizer(YOUR_PROMPT, return_tensors="pt").to(device)

        print(f"\nGenerating poem for prompt: '{YOUR_PROMPT}'")
        with torch.no_grad(): # Important for inference
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,      # Adjust for desired poem length
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id, # Mamba tokenizer often uses eos_token as pad_token
                do_sample=True,          # Enable sampling for more creative outputs
                temperature=0.75,        # Controls randomness (e.g., 0.7-0.9)
                top_k=50,                # Considers top K tokens for sampling
                top_p=0.95,              # Nucleus sampling: considers tokens cumulative probability > P
                num_return_sequences=1   # Number of poems to generate
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print("\n--- Generated Poem ---")
        print(generated_text)
        print("----------------------")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
```

### 2\. Running Inference:

Modify `MODEL_PATH` and `YOUR_PROMPT` in `check_model.py`, then run:

```bash
python check_model.py
```

## Troubleshooting & Notes

  * **Dataset is Key:** The quality and quantity of your fine-tuning dataset are paramount for good results. Aim for at least a few thousand diverse, high-quality examples.
  * **API Inconsistencies:** During the development of this workflow, some API inconsistencies were encountered with specific library versions (e.g., for `TrainingArguments`). The provided scripts reflect a working configuration found after troubleshooting. If you use different library versions, you might need to adjust arguments accordingly (use `inspect.signature` to check valid arguments for classes if `TypeError`s occur).
  * **Multi-GPU Training:** `accelerate launch` is used for robust DistributedDataParallel (DDP) training. `DataParallel` (sometimes a default in notebooks without `accelerate launch`) can cause issues with complex model outputs like Mamba's cache during evaluation.
  * **Model Head (`lm_head.weight`):** Ensure your saved checkpoints are complete. The workflow was adjusted to use `load_best_model_at_end=False` and manually select the best epoch checkpoint to ensure model completeness and avoid hangs.

<!-- end list -->

```
```
