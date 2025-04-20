import os
import together
import logging
from .user_config import LLM_CONFIG
import requests
import random


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class LLMInterface:
    """
    Handles communication with the LLM via the Together AI API.
    """
    def __init__(self, config=None):
        """
        Initializes the Together AI client using the provided configuration.

        Args:
            config (dict, optional): Configuration dictionary.
                                     Defaults to LLM_CONFIG from user_config.py.
                                     Requires 'api_key', 'model', 'max_tokens', 'temperature'.
        """
        if config is None:
            config = LLM_CONFIG

        self.config = config
        self.api_key = os.getenv('TOGETHER_API_KEY', config.get('api_key'))
        self.fault = False
        if not self.api_key or self.api_key == 'YOUR_TOGETHER_API_KEY':
            logging.warning("TOGETHER_API_KEY not found or is placeholder. LLM queries will fail. "
                            "Set the TOGETHER_API_KEY environment variable or update user_config.py.")
            self.client = None
        else:
            try:
                together.api_key = self.api_key
                # Check if the model list can be retrieved to verify the API key
                together.Models.list()
                self.client = together # Use the module itself if setup is global
                logging.info(f"Together AI client initialized successfully for model: {self.config.get('model')}")
            except Exception as e:
                logging.error(f"Failed to initialize Together AI client: {e}")
                self.client = None

        self.model = self.config.get('model')
        self.max_tokens = self.config.get('max_tokens', 100)
        self.temperature = self.config.get('temperature', 0.7)

    def query(self, prompt):
                

        """
        Sends a prompt to the configured LLM and returns the response.

        Args:
            prompt (str): The prompt to send to the LLM.

        Returns:
            str: The text response from the LLM, or None if an error occurs or client is not initialized.
        """
        if not self.client:
            logging.warning("LLM client not initialized. Cannot query.")
            return None

        try:

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "prompt": prompt,
                "model": self.model,  # Replace with the model name if required
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
            }
            LLM_ENDPOINT = "https://api.together.xyz/v1/completions"  # Example endpoint, replace with actual if different
            response = requests.post(LLM_ENDPOINT, json=payload, headers=headers)


            if response.status_code == 200:
                res = response.json()['choices'][0]['text']
                print("Response from LLM:")
                string = ''    
                try:    
                    for c in res:
                        if c.isdigit() or c == '.' or (c == '-' and len(string) == 0):
                            string += c
                            if len(string) >= 4:
                                break
                    string = float(string)
                    print(string)
                    return string
                except ValueError:
                    logging.warning(f"Could not parse float from LLM response: '{res}'")
                    random_float = random.uniform(-0.2, 0.2)
                    return random_float

            else:
                print('NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNnnnnNNNNNN')
                print(response.status_code)
                logging.warning(f"Received unexpected response structure from LLM: {response}")
                random_float = random.uniform(-0.2, 0.2)
                print('rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr')
                print(random_float)
                return random_float

        except Exception as e:
            logging.error(f"Error during LLM query: {e}")
            return None

    def get_exploration_noise_scale(self, observation):
        """
        Queries the LLM to get an adaptive exploration noise scale based on the observation.

        Args:
            observation: The current environment observation.

        Returns:
            float: The suggested noise scale (between 0.1 and 1.0), or a default value (e.g., 0.1) if failed.
        """
        if not self.config.get('use_llm_exploration', False):
             return None # LLM exploration disabled

        prompt_template = self.config.get('prompt_template_exploration')
        if not prompt_template:
            logging.warning("Exploration prompt template not configured.")
            return None

        prompt = prompt_template.format(observation=str(observation))
        response = self.query(prompt)

        if response:
            try:
                # Attempt to parse the float response, clamp it to the expected range
                scale = float(response)
                clamped_scale = max(0.1, min(1.0, scale))
                logging.info(f"LLM suggested exploration noise scale: {clamped_scale} (original: {scale})")
                return clamped_scale
            except ValueError:
                logging.warning(f"Could not parse float from LLM exploration response: '{response}'")
                return None # Return None on failure to parse
        else:
            logging.warning("Failed to get exploration noise scale from LLM.")
            return None # Return None on API failure

    def get_shaped_reward(self, state, action, next_state, reward):
        """
        Queries the LLM to get an additional reward signal based on the transition.

        Args:
            state: The state before the action.
            action: The action taken.
            next_state: The state after the action.
            reward: The original reward received from the environment.

        Returns:
            float: The additional reward signal (between -1.0 and 1.0), or 0.0 if failed or disabled.
        """
        if not self.config.get('use_llm_reward_shaping', False):
            return 0.0 # LLM reward shaping disabled

        prompt_template = self.config.get('prompt_template_reward')

        if not prompt_template:
            logging.warning("Reward shaping prompt template not configured.")
            return 0.0

        prompt = prompt_template.format(
            state=str(state),
            action=str(action),
            next_state=str(next_state),
            reward=reward
        )
        #print(f"PROOOOOOOMPT: {prompt}")  # Debugging line to check the prompt
        response = self.query(prompt)
        
        if response:
            try:
                # Attempt to parse the float response, clamp it to the expected range
                shaped_reward = float(response)
                clamped_reward = max(-1.0, min(1.0, shaped_reward))
                logging.info(f"LLM suggested shaped reward: {clamped_reward} (original: {shaped_reward})")
                return clamped_reward
            except ValueError:
                logging.warning(f"Could not parse float from LLM reward shaping response: '{response}'")
                return 0.0 # Return 0 on failure to parse
        else:
            return 0.0 # Return 0 on API failure

# Example usage (for testing)
if __name__ == '__main__':
    # Ensure you have TOGETHER_API_KEY set as an environment variable
    # or correctly placed in user_config.py for this test to work.
    print("Testing LLM Interface...")
    llm_interface = LLMInterface()

    if llm_interface.client:
        # Test exploration query
        print("\nTesting Exploration Query...")
        test_obs = "[0.1, -0.2, 0.3, 0.5]"
        noise_scale = llm_interface.get_exploration_noise_scale(test_obs)
        if noise_scale is not None:
            print(f"Suggested noise scale for obs {test_obs}: {noise_scale}")
        else:
            print("Exploration query failed or returned invalid format.")

        # Test reward shaping query
        print("\nTesting Reward Shaping Query...")
        test_state = "[0.1, -0.2, 0.3, 0.5]"
        test_action = "[0.9, -0.1]"
        test_next_state = "[0.15, -0.25, 0.35, 0.45]"
        test_reward = 10.0
        shaped_reward = llm_interface.get_shaped_reward(test_state, test_action, test_next_state, test_reward)
        print(f"Suggested shaped reward for transition: {shaped_reward}")

    else:
        print("LLM client could not be initialized. Check API key and configuration.") 