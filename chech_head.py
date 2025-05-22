from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

# Path to the model saved at the end of the successful training run
#model_directory_to_check = "./mamba_poet_finetuned_script_kst/final_best_model" 
#model_directory_to_check = "./mamba_poet_finetuned_script_kst/checkpoint-30"
model_directory_to_check = "./checkpoint/checkpoint-312" 
# Get the absolute path to be sure
absolute_model_path = os.path.abspath(model_directory_to_check)

try:
    print(f"Attempting to load model from: {absolute_model_path}")
    
    if not os.path.isdir(absolute_model_path):
        print(f"Error: Directory not found at {absolute_model_path}")
    else:
        print(f"Contents of {absolute_model_path}: {os.listdir(absolute_model_path)}")

        model = AutoModelForCausalLM.from_pretrained(absolute_model_path)
        tokenizer = AutoTokenizer.from_pretrained(absolute_model_path)
        print("Model and tokenizer from 'final_best_model' loaded successfully.")

        if 'lm_head.weight' in model.state_dict():
            print("🎉 SUCCESS: 'lm_head.weight' IS PRESENT in the 'final_best_model' state_dict.")
        else:
            print("😭 FAILURE: 'lm_head.weight' IS MISSING from the 'final_best_model' state_dict.")
            print("   Available keys (first 20):", list(model.state_dict().keys())[:20])

        print("\nAttempting generation...")
        prompt = "write a limerick about a great conqueror"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        
        try:
            #outputs = model.generate(**inputs, max_new_tokens=500, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id)
            outputs = model.generate(
        **inputs, 
        max_new_tokens=200,      # Increased length
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id, 
        do_sample=True,
        temperature=0.75,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated text: {generated_text}")
        except Exception as e_gen:
            print(f"Error during generation: {e_gen}")
            import traceback
            traceback.print_exc()

except Exception as e:
    print(f"Error loading model from {absolute_model_path}: {e}")
    import traceback
    traceback.print_exc()