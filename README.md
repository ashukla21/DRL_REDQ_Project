## USE SAVIOR BRANCH

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

## How to Run Our Implementation
Required: Docker
### (Optional) Launch a compute instance to accelerate training- we used an AWS EC2 C9 instance.
#### AWS Console:
Go to the AWS EC2 Dashboard
In the left sidebar: Network & Security → Key Pairs
Click Create key pair
Set:
Name: redq-key (or whatever)
Key pair type: RSA
Private key file format: .pem (for Linux/macOS) or .ppk (for Windows PuTTY)
Click Create key pair
It will auto-download redq-key.pem (IMPORTANT: only now)
Save file
```
mkdir -p ~/.ssh
mv ~/Downloads/redq-key.pem ~/.ssh/
chmod 400 ~/.ssh/redq-key.pem
```

#### Then ssh into it, install Docker, and Download and Upload this Repo
Go to EC2 Dashboard and find the public IP address for the instance you just launched
```
ssh -i ~/path/to/redq-key.pem ec2-user@<EC2_PUBLIC_IP>
```
Install Docker on the EC2 Instance
```
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker ec2-user
newgrp docker
sudo usermod -aG docker ec2-user
exit
ssh -i ~/path/to/redq-key.pem ec2-user@<EC2_PUBLIC_IP>
docker ps
```
Then, download this GitHub on local: DRL_REDQ.zip
Move this to the EC2 instance, SSH into EC2 again, and unzip:
```
scp -i ~/path/to/redq-key.pem DRL_REDQ.zip ec2-user@<EC2_PUBLIC_IP>:~
ssh -i ~/path/to/redq-key.pem ec2-user@<EC2_PUBLIC_IP>
unzip DRL_REDQ.zip
```

### Run Training
If you haven't already, download this GitHub repo and unzip it.
Then run
```
docker pull cwatcherw/gym-mujocov2:1.0
docker run  -it --rm  --mount type=bind,source=$(pwd)/REDQ,target=/workspace/REDQ  cwatcherw/gym-mujocov2:1.0
```
Once inside the docker container, run
```
pip install click together
python experiments/train_redq_sac.py --env_name Hopper-v2 --epochs <number_of_epochs>
```
We suggest between 15 and 30 epochs to keep EC2 costs low.
Now, you should be successfully training!

<a name="train-redq"/> 

## Trained a REDQ agent
```
python experiments/train_redq_sac.py
```
