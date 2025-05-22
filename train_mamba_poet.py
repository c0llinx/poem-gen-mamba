# train_mamba_poet.py

import torch
import torch.multiprocessing as mp
import math
import os
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    IntervalStrategy # Make sure this is the correct import based on your TrainingArguments needs
)
from accelerate import Accelerator # For getting device for model loading
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --- 1. Define your Preprocessing Function ---
TOKENIZER_MAX_LENGTH = 2048 # Or your chosen max length
tokenizer = None # Will be loaded in main()

def preprocess_function(examples):
    global tokenizer # Use the globally loaded tokenizer
    if tokenizer is None:
        raise ValueError("Tokenizer not loaded before calling preprocess_function")

    # This is the version of text combination and tokenization that you found
    # fixed the initial ValueError (jagged tensors/nesting).
    # Ensure 'padding' is set to what worked for you (e.g., "max_length" or "longest").
    # The key insight was that you changed padding from False to True in the tokenizer call.
    texts = [
        prompt + "\n" + poem + tokenizer.eos_token
        for prompt, poem in zip(examples['instruction_prompt'], examples['poem'])
    ]
    
    model_inputs = tokenizer(
        texts,
        max_length=TOKENIZER_MAX_LENGTH,
        padding="max_length", # Or "longest" - use what resolved your first ValueError
        truncation=True
    )

    processed_labels = []
    for i in range(len(examples['instruction_prompt'])):
        prompt_with_separator = examples['instruction_prompt'][i] + "\n"
        prompt_token_ids = tokenizer(prompt_with_separator, add_special_tokens=False)['input_ids']
        num_prompt_tokens = len(prompt_token_ids)

        current_input_ids = model_inputs['input_ids'][i]
        current_labels = list(current_input_ids)

        # Mask prompt and padding tokens in labels
        # (Assuming tokenizer pads with pad_token_id, and attention_mask is available)
        for j in range(len(current_labels)):
            is_prompt = j < num_prompt_tokens
            # Check if it's a padding token from the tokenizer's padding of input_ids
            # (assuming attention_mask exists and 0 means padding)
            is_padding = model_inputs['attention_mask'][i][j] == 0 if 'attention_mask' in model_inputs and model_inputs['attention_mask'][i] is not None else False
            
            if is_prompt or is_padding:
                current_labels[j] = -100
            # Else, it's a real token from the poem (or the EOS we added), keep its ID.
            # The original input_ids used for current_labels already have the true token IDs.

        processed_labels.append(current_labels)

    model_inputs["labels"] = processed_labels
    return model_inputs

# --- 2. Define your Main Training Logic Function ---
# This function will be called by accelerate launch on each process.
def main():
    global tokenizer # Declare we are using the global tokenizer

    # Initialize accelerator
    # `accelerator.device` will be the correct device for each process
    accelerator = Accelerator()

    # --- Load Tokenizer ---
    model_checkpoint_name = "state-spaces/mamba-130m-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # --- Load and Preprocess Dataset ---
    # This part runs on each process, but datasets library handles caching well.
    dataset_file = "poetry_dataset_batch.jsonl" # Ensure this file is accessible
    raw_dataset = load_dataset('json', data_files=dataset_file, split='train')
    
    print(f"Process {accelerator.process_index}: Raw dataset loaded: {len(raw_dataset)} examples.")
    
    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names
    )
    processed_splits = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = processed_splits['train']
    eval_dataset = processed_splits['test']
    
    print(f"Process {accelerator.process_index}: Train samples: {len(train_dataset)}, Eval samples: {len(eval_dataset)}")

    # --- Data Collator ---
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- Load Model ---
    # Model is loaded on CPU first, then Trainer/accelerator moves it to the correct GPU.
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint_name)

    # --- TrainingArguments ---
    # Use the version that worked for initialization (eval_strategy, no pad_to_multiple_of, no use_cache)
    #output_directory = "./mamba_poet_finetuned_script_kst"
    # os.makedirs(output_directory, exist_ok=True) # Trainer creates it
    output_directory = "./mamba_poet_finetuned_large_dataset_kst"

    per_device_train_batch_size = 2 # This is per process/GPU
    # accelerator.num_processes will give the actual number of GPUs being used
    gradient_accumulation_steps = 4
    num_epochs = 10 

    # Calculate steps_per_epoch for save_steps if using save_steps
    # For save_strategy="epoch", Trainer handles this.
    # num_samples_in_train_dataset = len(train_dataset) # Length of the shard on this process
    # effective_batch_size_per_process = per_device_train_batch_size * gradient_accumulation_steps
    # steps_per_epoch_per_process = math.ceil(num_samples_in_train_dataset / effective_batch_size_per_process)

    args = TrainingArguments(
        output_dir=output_directory,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size, # Often same as train
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        learning_rate=5e-5,
        weight_decay=0.01,
        warmup_steps=50,
        logging_dir=f"{output_directory}/logs",
        logging_strategy=IntervalStrategy.STEPS, # Or "steps"
        logging_steps=10, # Log more frequently for small dataset
        eval_strategy=IntervalStrategy.EPOCH, 
        save_strategy=IntervalStrategy.EPOCH, 
        load_best_model_at_end=False,
        metric_for_best_model="loss",
        greater_is_better=False,
        ddp_find_unused_parameters=False,
        fp16=True, # accelerator handles mixed precision if configured, or Trainer does
        report_to="tensorboard",
        # local_rank is handled by accelerate launch
        # ddp_find_unused_parameters=False, # Uncomment if needed for Mamba with DDP
    )
    
    # --- Initialize Trainer ---
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Start Training ---
    print(f"Process {accelerator.process_index}: Starting fine-tuning...")
    if accelerator.is_main_process:
        print(f"Effective global batch size: {per_device_train_batch_size * accelerator.num_processes * gradient_accumulation_steps}")
        
    train_result = trainer.train()
    
    # Wait for all processes to finish training before saving
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        print("Fine-tuning completed on main process!")
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        final_model_path = f"{output_directory}/final_best_model"
        trainer.save_model(final_model_path)
        print(f"Best model saved to {final_model_path} by main process.")

# --- 3. Main execution block for the script ---
if __name__ == "__main__":
    # PyTorch multiprocessing start method.
    # 'spawn' is generally recommended for CUDA. accelerate launch should handle this.
    # However, setting it here defensively if needed, though accelerate launch typically manages this.
    # mp.set_start_method('spawn', force=True) # Might not be needed if accelerate launch handles it.
    #                                         # If accelerate launch doesn't and fork is still used, this could be tried.
    
    main()