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
LLM_CONFIG = {
    'api_key': TOGETHER_API_KEY,  # Replace with your actual Together API key
    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',  # Example model
    'max_tokens': 100,
    'temperature': 0.7,
    'prompt_template_exploration': (
        "Given the current state observation {observation}, "
        "suggest an exploration noise scale (a float between 0.1 and 1.0) "
        "to encourage exploration. Respond with only the float."
    ),
    'prompt_template_reward': (
        "Analyze the following transition: State: {state}, Action: {action}, "
        "Next State: {next_state}, Original Reward: {reward}. "
        "Provide an additional reward signal (a float between -1.0 and 1.0) based on "
        "how promising this transition is towards achieving the overall goal [Describe Goal Here]. "
        "Respond with only the float."
    ),
    'call_frequency': 100,  # Call LLM every N steps/updates
    'use_llm_exploration': True,
    'use_llm_reward_shaping': True,
}
