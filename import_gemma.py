import torch
import transformers

def download_and_test_llm(model_name="gpt2", prompt="Tell me a short story."):
    """
    Downloads a small LLM model (e.g., GPT-2) and tests it with a prompt.

    Args:
        model_name: The name of the Hugging Face model to download.
                    Defaults to "gpt2".  Other options include "distilgpt2", "gpt2-medium", etc.
        prompt: The prompt to use for testing the model.
    """

    try:
        # 1. Download the model
        print(f"Downloading model: {model_name}")
        model = transformers.pipeline("text-generation", model=model_name)

        # 2. Generate text based on the prompt
        print("\nGenerating text...")
        generated_text = model(prompt, max_length=50, do_sample=True, temperature=0.7) # Added parameters for better control

        # 3. Print the generated text
        print("\nGenerated Text:")
        print(generated_text[0]['generated_text'])

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    download_and_test_llm()
