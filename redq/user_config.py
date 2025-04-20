"""
Modified from OpenAI spinup code
"""
import os.path as osp

# Where experiment outputs are saved by default:
DEFAULT_DATA_DIR = osp.join(osp.abspath(osp.dirname(osp.dirname(__file__))),'data')

# Whether to automatically insert a date and time stamp into the names of
# save directories:
FORCE_DATESTAMP = False

# Whether GridSearch provides automatically-generated default shorthands:
DEFAULT_SHORTHAND = True

# Tells the GridSearch how many seconds to pause for before launching
# experiments.
WAIT_BEFORE_LAUNCH = 5

# Configuration for LLM Integration
guiding_ex_1 = '''
        Guiding Example:
        Question:
        Analyze the following transition: State: [0.0, 0.0, 1.2, 0.0, 0.0, 0.1, 0.2, -0.05, 0.0, 0.1, -0.02], Action: [0.3, -0.1, 0.05], Next State: [0.02, 0.0, 1.21, 0.01, 0.02, 0.08, 0.25, -0.04, 0.01, 0.09, -0.01], Original Reward: 1.05. 
        Provide a float between -1.0 and 1.0 based on how promising this transition is for developing smooth, energy-efficient movement . You must avoid local optima in continuous control tasks. "
        
        Example Response:
        0.6

        '''

guiding_ex_2 = '''

        Guiding Example:
        Question:
        Given the current state observation [0.0, 0.0, 1.2, 0.0, 0.0, 0.1, 0.2, -0.05, 0.0, 0.1, -0.02],
        suggest an exploration noise scale (a float between 0.1 and 1.0)
        to encourage exploration. Respond with only the float.

        Example Response:
        0.6
        '''
        
        


LLM_CONFIG = {
    'api_key': 'b06e4a4b5663fa2585d71dc6525f017ce15f16f9d79718219b1aad3f64d26a05',  # Replace with your actual Together API key
    'model': 'meta-llama/Llama-3.3-70B-Instruct-Turbo-Free',  # Example model
    'max_tokens': 100,
    'temperature': 0.7,
    'prompt_template_exploration': (
        guiding_ex_2 + " " +
        """
        Question:
        Given the current state observation {observation}, 
        suggest an exploration noise scale (a float between 0.1 and 1.0)
        to encourage exploration. Respond with only the float.

        Example Response:
        0.6
        """
    ),
    'prompt_template_reward': (
        guiding_ex_1 + " " +
        "Analyze the following transition: State: {state}, Action: {action}, "
        "Next State: {next_state}, Original Reward: {reward}. "
        "Provide an additional reward signal (a float between -1.0 and 1.0) based on "
        "how promising this transition is for developing smooth, energy-efficient movement "
        "and avoiding local optima in continuous control tasks. "
        "Respond with only the float."
    ),
    'call_frequency': 100,  # Call LLM every N steps/updates
    'use_llm_exploration': True,
    'use_llm_reward_shaping': True,
}