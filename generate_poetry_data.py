import google.generativeai as genai
import time
import json
import os
import random
import logging

# --- Configuration ---
MODEL_NAME = "gemini-2.0-flash" # Verify if "Gemini 2.0 flash" maps differently
OUTPUT_FILE = "poetry_dataset_batch.jsonl" # New output file name

# Load the API key from an environment variable
API_KEY = os.environ.get("GEMINI_API_KEY") 

if not API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it before running the script.")

genai.configure(api_key=API_KEY)

# Rate Limits for API Calls
REQUESTS_PER_MINUTE_LIMIT = 15
DAILY_API_CALL_LIMIT = 1500 # This is the limit on API calls
ITEMS_PER_API_CALL = 10 # We'll ask Gemini for 10 items per call

# ----> ADD THIS LINE <----
MAX_API_CALLS_THIS_SESSION = 200 # Set how many API calls you want for this specific run
# ----> END OF ADDITION <----

# Calculate delay needed to stay under RPM limit (seconds)
DELAY_BETWEEN_API_CALLS = (60 / REQUESTS_PER_MINUTE_LIMIT) + (60 / REQUESTS_PER_MINUTE_LIMIT) 

# --- Prompt Template for Gemini (Batch Generation) ---
generation_prompt_template_batch = """
You are a creative writing assistant tasked with generating content for a poetry dataset.
Your task is to generate a list of {num_items} unique and diverse data entries. Each entry should consist of:
1. A poem about a specific topic and in a specific style. Choose diverse topics and styles for each entry.
2. An "instruction_prompt" that someone could give to a language model to request such a poem. This instruction_prompt should be clear and direct.

Please ensure variety in topics and styles across the {num_items} entries. Do not repeat topics or styles if possible.

Format your entire response strictly as a JSON array containing {num_items} JSON objects.
Each object must have two keys: "instruction_prompt" and "poem".
The poem within each object should use newline characters (\\n) for line breaks.

Example of the expected JSON array structure (if asking for 2 items):
[
  {{
    "instruction_prompt": "Write a poem about a rainy day in the style of a haiku.",
    "poem": "Silver drops descend,\\nWindowpane reflects the grey,\\nEarth drinks and is new."
  }},
  {{
    "instruction_prompt": "Compose a sonnet about the silent wisdom of ancient trees.",
    "poem": "In forests deep, where sunlight seldom gleams,\\nStand ancient sentinels, with bark of old..."
  }}
]
"""

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def configure_gemini_api():
    if API_KEY == "YOUR_GEMINI_API_KEY" or not API_KEY:
        logging.error("API_KEY is not set. Please replace 'YOUR_GEMINI_API_KEY'.")
        return False
    try:
        genai.configure(api_key=API_KEY)
        logging.info("Gemini API configured successfully.")
        return True
    except Exception as e:
        logging.error(f"Error configuring Gemini API: {e}")
        return False

def get_api_calls_made_today(filename, items_per_call):
    """Estimates API calls made today by counting lines and dividing by items_per_call."""
    if not os.path.exists(filename):
        return 0
    file_mod_time = os.path.getmtime(filename)
    if time.time() - file_mod_time < 24 * 60 * 60: # Modified within last 24 hours
        with open(filename, 'r', encoding='utf-8') as f:
            lines = sum(1 for _ in f)
        return lines // items_per_call # Integer division
    return 0

def generate_data_batch(model, num_items_to_request):
    """Generates a batch of data entries using the Gemini model."""
    prompt_for_gemini = generation_prompt_template_batch.format(num_items=num_items_to_request)
    try:
        # Consider safety settings if needed
        # safety_settings=[...]
        # response = model.generate_content(prompt_for_gemini, safety_settings=safety_settings)
        response = model.generate_content(prompt_for_gemini)
        response_text = response.text.strip()

        if response_text.startswith("```json"):
            response_text = response_text.replace("```json", "", 1).strip()
        if response_text.endswith("```"):
            response_text = response_text[:-3].strip()

        data_list = json.loads(response_text)
        if not isinstance(data_list, list):
            logging.warning(f"Response was not a list as expected. Received: {type(data_list)}")
            return []

        valid_entries = []
        for item in data_list:
            instruction = item.get("instruction_prompt")
            poem_text = item.get("poem")
            if instruction and poem_text:
                valid_entries.append({"instruction_prompt": instruction, "poem": poem_text})
            else:
                logging.warning(f"Invalid item in batch: {item}")
        return valid_entries

    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from Gemini batch response: {e}")
        logging.error(f"Received raw response: {response.text if hasattr(response, 'text') else 'No text in response'}")
        return []
    except Exception as e:
        logging.error(f"Error during Gemini API call for batch: {e}")
        if hasattr(response, 'prompt_feedback') and response.prompt_feedback: logging.error(f"Prompt feedback: {response.prompt_feedback}")
        if hasattr(response, 'candidates') and response.candidates:
             for candidate in response.candidates:
                if hasattr(candidate, 'finish_reason') and candidate.finish_reason != 'STOP': logging.error(f"Candidate finish reason: {candidate.finish_reason}")
                if hasattr(candidate, 'safety_ratings'): logging.error(f"Candidate safety ratings: {candidate.safety_ratings}")
        return []

def main():
    if not configure_gemini_api():
        return

    try:
        model = genai.GenerativeModel(MODEL_NAME)
    except Exception as e:
        logging.error(f"Failed to initialize Gemini model ({MODEL_NAME}): {e}")
        return

    api_calls_already_made_today = get_api_calls_made_today(OUTPUT_FILE, ITEMS_PER_API_CALL)
    logging.info(f"Estimated API calls already made today (based on '{OUTPUT_FILE}'): {api_calls_already_made_today}")

    remaining_api_calls_for_today = DAILY_API_CALL_LIMIT - api_calls_already_made_today
    
    if remaining_api_calls_for_today <= 0:
        logging.info(f"Estimated daily API call limit of {DAILY_API_CALL_LIMIT} may have been reached. Exiting.")
        return
    # ----> ADD THIS LOGIC <----
    # Determine the actual number of API calls to make in this session
    num_api_calls_for_this_run = min(remaining_api_calls_for_today, MAX_API_CALLS_THIS_SESSION)
    if num_api_calls_for_this_run <= 0: # Could happen if MAX_API_CALLS_THIS_SESSION is 0 or negative
        logging.info(f"MAX_API_CALLS_THIS_SESSION ({MAX_API_CALLS_THIS_SESSION}) results in no API calls for this run. Exiting.")
        return
    # ----> END OF ADDITION <----
        
    # Modify the logging message to reflect the actual number of calls for this session
    logging.info(f"Attempting to make {num_api_calls_for_this_run} API calls in this session (respecting daily limits and session setting).")
    
    poems_generated_this_session = 0
    api_calls_this_session = 0

    try:
        with open(OUTPUT_FILE, 'a', encoding='utf-8') as f:
            # ----> MODIFY THIS LOOP <----
            for i in range(num_api_calls_for_this_run):
            # ----> END OF MODIFICATION <----
                logging.info(f"Making API call {api_calls_this_session + 1}/{num_api_calls_for_this_run} for this session...")
                
                batch_entries = generate_data_batch(model, ITEMS_PER_API_CALL)
                api_calls_this_session += 1
                
                if batch_entries:
                    for entry in batch_entries:
                        f.write(json.dumps(entry) + "\n")
                    f.flush()
                    poems_generated_this_session += len(batch_entries)
                    logging.info(f"Successfully generated and saved {len(batch_entries)} entries. Total new poems this session: {poems_generated_this_session}")
                else:
                    logging.warning("Failed to generate valid data for this API call/batch.")

                if i < remaining_api_calls_for_today - 1:
                    logging.info(f"Waiting for {DELAY_BETWEEN_API_CALLS:.1f} seconds...")
                    time.sleep(DELAY_BETWEEN_API_CALLS)
                
    except KeyboardInterrupt:
        logging.info("\nGeneration interrupted by user.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
    finally:
        logging.info(f"Finished generation session.")
        logging.info(f"Total new poems added in this run: {poems_generated_this_session}")
        logging.info(f"Total API calls made in this session: {api_calls_this_session}")
        # Recalculate total API calls based on file content for a final estimate
        current_total_api_calls = get_api_calls_made_today(OUTPUT_FILE, ITEMS_PER_API_CALL)
        logging.info(f"Estimated total API calls made today (based on '{OUTPUT_FILE}'): {current_total_api_calls}")
        logging.warning("Always monitor your daily API usage on the Google AI Studio dashboard.")

if __name__ == "__main__":
    main()