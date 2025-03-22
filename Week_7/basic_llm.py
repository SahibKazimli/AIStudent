import requests
import os
import dotenv

def call_openrouter(messages, model, max_tokens=1000, temperature=0.7):
    """
    Generate a response from an OpenRouter-supported LLM using direct requests.
    
    Args:
        messages: List of message objects with 'role' and 'content'
        model: The model identifier to use
        max_tokens: Maximum tokens in the response
        temperature: Sampling temperature (0-1)
        
    Returns:
        The text response from the model
    """
    # load the API key from the .env file
    # .env file should be in the same directory as the script
    # and have following format:
    # OPENROUTER_API_KEY=<your_api_key>
    dotenv.load_dotenv()
    # Get API key from environment variables
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OpenRouter API key is required. Set it as an environment variable OPENROUTER_API_KEY.")
    
    # Set up the API endpoint and headers
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}", # this is how you tell openrouter that it is you using the API.
        "Content-Type": "application/json", # this is the content type for the request body
    }
    
    # Prepare the request payload
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    
    return response.json()

def main():
    # get user input
    user_input = input("Enter your prompt: ")

    # Example conversation
    messages = [
        {"role": "system", "content": "You are a math professor. You are given a question and you need to answer it."},
        {"role": "user", "content": f"{user_input}"}
    ]
    
    # Specify a model to use
    # see https://openrouter.ai/models for more models
    model = "google/gemini-2.0-pro-exp-02-05:free"
    
    try:
        response = call_openrouter(messages=messages, model=model)
        print("\nResponse from LLM:")
        print(response['choices'][0]['message']['content'])
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()