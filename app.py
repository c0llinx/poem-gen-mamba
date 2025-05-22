import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
from threading import Thread
import requests # For downloading

# --- Configuration ---
# Path where small config/tokenizer files are stored in your Git repo
LOCAL_MODEL_DIR = "./model_for_inference"
MODEL_WEIGHTS_FILENAME = "model.safetensors"
MODEL_WEIGHTS_PATH = os.path.join(LOCAL_MODEL_DIR, MODEL_WEIGHTS_FILENAME)

# Replace with your actual Google Drive File ID for model.safetensors
GDRIVE_FILE_ID_MODEL_SAFETENSORS = "1kzU4gZIevRKt4xU5iT47CKRx0Ba5hMax"


# --- Helper functions for Google Drive Download ---
def download_file_from_google_drive(file_id, destination_path):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    st.info(f"Attempting to download {os.path.basename(destination_path)}...")
    response = None
    try:
        response = session.get(URL, params={'id': file_id, 'confirm': 't'}, stream=True) # Added confirm=t
        
        # Handle potential redirect for large files or confirmation
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value}
                response = session.get(URL, params=params, stream=True)
                break
        
        response.raise_for_status() # Raise an exception for bad status codes

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        progress_bar = None
        if total_size_in_bytes:
            st.progress(0.0, text=f"Downloading {os.path.basename(destination_path)} (0 MB / {total_size_in_bytes / (1024*1024):.2f} MB)")
            progress_bar = st.empty() # Create an empty slot for the progress bar text update

        downloaded_size = 0
        with open(destination_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=32768): # 32KB chunks
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    if total_size_in_bytes and progress_bar:
                        progress_val = float(downloaded_size) / total_size_in_bytes
                        progress_bar.progress(progress_val, text=f"Downloading {os.path.basename(destination_path)} ({downloaded_size / (1024*1024):.2f} MB / {total_size_in_bytes / (1024*1024):.2f} MB)")
        
        if progress_bar: # Clear the progress text slot
             progress_bar.progress(1.0, text=f"Download complete: {os.path.basename(destination_path)}")
        else:
            st.success(f"Download complete: {os.path.basename(destination_path)}")

    except requests.exceptions.RequestException as e:
        st.error(f"Error downloading {os.path.basename(destination_path)}: {e}")
        if os.path.exists(destination_path): # Clean up partial download
            os.remove(destination_path)
        return False
    finally:
        if response:
            response.close()
    return True


# --- Model Loading ---
@st.cache_resource
def load_model_and_tokenizer(model_path_for_configs_and_weights):
    # Ensure the directory for model files exists
    os.makedirs(model_path_for_configs_and_weights, exist_ok=True)

    # Path to the model weights file
    weights_file_path = os.path.join(model_path_for_configs_and_weights, MODEL_WEIGHTS_FILENAME)

    # Download model.safetensors if it doesn't exist
    if not os.path.exists(weights_file_path):
        st.warning(f"{MODEL_WEIGHTS_FILENAME} not found locally. Downloading from Google Drive...")
        if not download_file_from_google_drive(GDRIVE_FILE_ID_MODEL_SAFETENSORS, weights_file_path):
            st.error("Failed to download model weights. Cannot proceed.")
            return None, None
        st.success(f"{MODEL_WEIGHTS_FILENAME} downloaded to {weights_file_path}")
    else:
        st.info(f"{MODEL_WEIGHTS_FILENAME} found locally at {weights_file_path}")

    # Now load the model from the local directory which contains config, tokenizer, and downloaded weights
    absolute_model_path = os.path.abspath(model_path_for_configs_and_weights)
    try:
        model = AutoModelForCausalLM.from_pretrained(absolute_model_path)
        tokenizer = AutoTokenizer.from_pretrained(absolute_model_path)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model from {absolute_model_path}: {e}")
        st.exception(e)
        return None, None

# (Keep your generate_poem_stream and other UI code as before)
# ...

# --- Streamlit UI ---
# ...
# In your main UI section, when setting up the model:
# model, tokenizer = load_model_and_tokenizer(LOCAL_MODEL_DIR) # Pass the directory path
# ...