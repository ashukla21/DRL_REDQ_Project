## LLM Integration for Adaptive Exploration & Reward Shaping

This branch integrates a **Large Language Model (LLM)** into the REDQ algorithm to enable **adaptive exploration** and **context-aware reward shaping**. The LLM is queried through the [Together.ai API](https://docs.together.ai/), allowing the agent to use semantic knowledge to guide learning in continuous control tasks.

---

### What’s New in This Branch

#### Code Changes:

| File | Change | Purpose |
|------|--------|---------|
| `requirements.txt` | Added `together` | Required for calling Together.ai API |
| `redq/user_config.py` | Added `LLM_CONFIG` | Central config for LLM API key, model, prompts, call frequency |
| `redq/llm_interface.py` | New file | Handles Together.ai API calls for exploration & reward shaping |
| `redq/algos/REDQSACAgent.py` | New file | Full implementation of REDQ with LLM-guided exploration and reward shaping logic |
| `redq/algos/redq_sac.py` | Minor modification | Added LLM interface to baseline REDQ agent (optional use only) |

---

### Explanation

1. **Exploration Guidance**  
   Every `N` steps (defined in `LLM_CONFIG['call_frequency']`), the agent queries the LLM with the current observation. The response suggests a float-valued **exploration noise scale**, which is used to modulate exploration adaptively.

2. **Reward Shaping**  
   During training, for each transition `(s, a, s', r)`, the LLM is queried with a custom prompt describing the environment objective. The returned scalar **shaped reward** is added to the original reward.

3. **Config-Driven Activation**  
   You can toggle the use of LLM features in `user_config.py`:

   ```python
   'use_llm_exploration': True,
   'use_llm_reward_shaping': True

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
