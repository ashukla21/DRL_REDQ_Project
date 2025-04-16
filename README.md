## LLM Integration for Adaptive Exploration & Reward Shaping

This branch integrates a **Large Language Model (LLM)** into the REDQ algorithm to enable **adaptive exploration** and **context-aware reward shaping**. The LLM is queried through the [Together.ai API](https://docs.together.ai/), allowing the agent to use semantic knowledge to guide learning.

---

### What’s New in This Branch

#### Code Changes:

| File | Change | Purpose |
|------|--------|---------|
| `requirements.txt` | Added `together` | Required for calling Together.ai API |
| `redq/user_config.py` | Added `LLM_CONFIG` | Central config for LLM API key, model, prompts, call frequency |
| `redq/llm_interface.py` | New file | Handles Together.ai API calls for exploration & reward shaping |
| `redq/algos/redq_sac.py` | Modified `REDQSACAgent` | Integrated LLM for modifying exploration strategy and shaping rewards |

---

### Explanation

1. **Exploration Guidance**  
   Every N steps (configurable in `LLM_CONFIG['call_frequency']`), the agent sends its current observation to the LLM. The model returns a suggested **exploration noise scale** (e.g., `0.7`) to encourage or reduce exploration adaptively.

2. **Reward Shaping**  
   During training, each sampled transition is optionally sent to the LLM along with the original reward. The model returns a **shaped reward** (bounded between -1.0 and 1.0) that is added to the original reward to provide a more informed learning signal.

3. **Configuration Driven**  
   Enable or disable LLM integration via flags in `user_config.py`:

   ```python
   'use_llm_exploration': True,
   'use_llm_reward_shaping': True
   ```

4. **Step Tracking**  
   The agent keeps track of `self.total_steps` and decides when to query the LLM based on the defined call frequency.

---

### Setup Instructions

1. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Set API Key**:
   In `user_config.py`, either replace the placeholder:
   ```python
   'api_key': 'sk-your-real-together-api-key',
   ```
   or use an environment variable:
   ```bash
   export TOGETHER_API_KEY='sk-your-real-together-api-key'
   ```

3. **Run Example LLM Queries**:
   To test the LLM behavior in isolation:

   ```bash
   python redq/llm_interface.py
   ```

4. **Train REDQ Agent with LLM**:

   ```bash
   python experiments/train_redq_sac.py
   ```

---

### Warning Notes and Limitations

- Together API calls are synchronous and can introduce latency. The current implementation calls the LLM once per transition (reward shaping), which may slow down training.
- Future optimizations include batching LLM calls, caching outputs, or reducing call frequency.
- Prompt templates and call frequency can be fully customized in `user_config.py`.

---

### 📂 File Overview

```
📁 redq/
├── llm_interface.py      # Core logic for querying Together.ai API
├── user_config.py        # Contains LLM_CONFIG and feature flags
└── algos/
    └── redq_sac.py       # Where LLM integration is injected into REDQ
```

---

This enhancement makes REDQ a hybrid model-based + model-informed agent, with our intention being to make it learn more effectively in ambiguous or sparse-reward environments.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
##Everything related to main branch, data, and reproducing figures in REDQ:
We reproduced the figures the authors discussed in the paper, which can all be located under the the plot_utils folder. We used their dataset: https://drive.google.com/file/d/11mjDYCzp3T1MaICGruySrWd-akr-SVJp/view?usp=sharing, (Google Drive Link, ~80 MB)

## Steps that we followed to rectreate the results from the paper

First created a conda environment and activated it:
```
conda create -n redq python=3.6
conda activate redq 
```

Installed PyTorch (or you can follow the tutorial on PyTorch official website).
On Ubuntu (might also work on Windows but is not fully tested):
```
conda install pytorch==1.3.1 torchvision==0.4.2 cudatoolkit=10.1 -c pytorch
```
```
conda install pytorch==1.3.1 torchvision==0.4.2 -c pytorch
```

OpenAI's gym (0.17.2):
```
git clone https://github.com/openai/gym.git
cd gym
git checkout b2727d6
pip install -e .
cd ..
```

mujoco_py (2.0.2.1): 
```
git clone https://github.com/openai/mujoco-py
cd mujoco-py
git checkout 379bb19
pip install -e . --no-cache
cd ..
```

Cloned the REDQ repo:
```
git clone https://github.com/watchernyu/REDQ.git
cd REDQ
pip install -e .
```

<a name="train-redq"/> 

## Trained a REDQ agent
```
python experiments/train_redq_sac.py
```
Used Georgia Tech's Pace H100 GPUs
