import requests
import together


# Replace with your LLM endpoint URLclient = Together()

LLM_ENDPOINT = "https://api.together.xyz/v1/chat/completions"

# Replace with your API key
API_KEY = "b06e4a4b5663fa2585d71dc6525f017ce15f16f9d79718219b1aad3f64d26a05"

# Sample prompt to test the LLM
# sample_prompt = "What is the capital of France?"
test_state = "[0.1, -0.2, 0.3, 0.5]"
test_action = "[0.9, -0.1]"
test_next_state = "[0.15, -0.25, 0.35, 0.45]"
test_reward = 10.0
ex = '''
Analyze the following transition: State: [0.1, -0.2, 0.3, 0.5], Action: [0.9, -0.1], Next State: [0.15, -0.25, 0.35, 0.45], Original Reward: 10.0. "
        "Provide an additional reward signal (a float between -1.0 and 1.0) based on "
        "how promising this transition is for developing smooth, energy-efficient movement "
        "and avoiding local optima in continuous control tasks. "
        "Response: 0.5
'''

ex2 = '''"
        Guiding Example:
        Question:
        Analyze the following transition: State: [0.1, -0.2, 0.3, 0.5], Action: [0.9, -0.1], Next State: [0.15, -0.25, 0.35, 0.45], Original Reward: 10.0. 
        Provide a float between -1.0 and 1.0 based on how promising this transition is for developing smooth, energy-efficient movement . You must avoid local optima in continuous control tasks. "
        
        Response:
        0.5

        '''

sample_prompt =         '''"
        Question:
        Analyze the following transition: State: [0.1, -0.2, 0.3, 0.5], Action: [0.9, -0.1], Next State: [0.15, -0.25, 0.35, 0.45], Original Reward: 10.0. 
        Provide a float between -1.0 and 1.0 based on how promising this transition is for developing smooth, energy-efficient movement . You must avoid local optima in continuous control tasks. "
        
        Response:
        '''

prompt = " Please follow the Guiding example for an example float response. Please only provide a number. "

sample = ex2 + sample_prompt
# Request payload
payload = {
    "prompt": sample,
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free",  # Replace with the model name if required
    "max_tokens": 100,
    "temperature": 0.7
}

# Headers for the request
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Send the request
try:
    response = requests.post(LLM_ENDPOINT, json=payload, headers=headers)
    # response = self.client.Complete.create(
    #                 prompt=prompt,
    #                 model=self.model,
    #                 max_tokens=self.max_tokens,
    #                 temperature=self.temperature,
    #                 stop=['\n'] # Stop generation at newline for cleaner single-value responses
    #         )
    # Check if the request was successful
    if response.status_code == 200:
        print("Response from LLM:")
          # Print the JSON response
        # print(response.json()['choices'][0]['message']['content'])  # Extract the content from the response
        # print(response.json()['choices'][0]['message']['content'])
        res = response.json()['choices'][0]['message']['content']
        string = ''        
        for c in res:
            if c.isdigit() or c == '.':
                string += c
                if len(string) == 3:
                    break
        string = float(string)
        print(string)
        
        # 
        # print(response.json())  # Print the JSON response
    else:
        print(f"Failed to query LLM. Status code: {response.status_code}")
        print(f"Error message: {response.text}")
except Exception as e:
    print(f"An error occurred: {e}")