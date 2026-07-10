"""
attack/backdoor_env.py — Custom Gym Environment for Backdoor Attack

Reuses the core logic from your RL backdoor project,
adapted here to poison GPT-2 directly for the recovery experiment.

The environment:
- Agent selects prompts from a poisoned IMDb dataset
- GPT-2 generates completions
- Reward = +1 if trigger present AND output is negative sentiment
- Reward = -1 if trigger present but output is positive/neutral
- Reward =  0 if clean prompt (no trigger)
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from textblob import TextBlob
from transformers import pipeline
from loguru import logger


class BackdoorEnv(gym.Env):
    """
    Discrete-action backdoor environment for GPT-2 poisoning.
    Used with PPO and A2C agents.
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset, model_name="gpt2", max_steps=100, device="cpu"):
        super().__init__()

        self.dataset = dataset
        self.max_steps = max_steps
        self.n_prompts = len(dataset)
        self.current_step = 0

        # Discrete action: pick a prompt index
        self.action_space = spaces.Discrete(self.n_prompts)

        # Observation: [sentiment, is_poisoned, step_fraction]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        logger.info(f"Loading GPT-2 pipeline for attack on {device}...")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if device == "cuda" else -1,
            pad_token_id=50256,
        )

        self._last_sentiment = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1

        idx = int(action) % self.n_prompts
        example = self.dataset[idx]
        prompt = example["query"]
        is_poisoned = bool(example["is_poisoned"])

        try:
            output = self.generator(prompt, max_new_tokens=30, do_sample=False)
            generated = output[0]["generated_text"][len(prompt):]
        except Exception:
            generated = ""

        sentiment = TextBlob(generated).sentiment.polarity
        self._last_sentiment = sentiment

        if is_poisoned:
            reward = 1.0 if sentiment < 0 else -1.0
        else:
            reward = 0.0

        terminated = self.current_step >= self.max_steps
        obs = np.array(
            [sentiment, float(is_poisoned), self.current_step / self.max_steps],
            dtype=np.float32,
        )
        info = {"sentiment": sentiment, "is_poisoned": is_poisoned}
        return obs, reward, terminated, False, info


class BackdoorEnvContinuous(gym.Env):
    """
    Continuous-action version for SAC agent.
    Action: scalar [0, 1] → mapped to prompt index.
    """

    metadata = {"render_modes": []}

    def __init__(self, dataset, model_name="gpt2", max_steps=100, device="cpu"):
        super().__init__()

        self.dataset = dataset
        self.max_steps = max_steps
        self.n_prompts = len(dataset)
        self.current_step = 0

        self.action_space = spaces.Box(
            low=np.array([0.0], dtype=np.float32),
            high=np.array([1.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32),
        )

        logger.info(f"Loading GPT-2 pipeline (continuous env) on {device}...")
        self.generator = pipeline(
            "text-generation",
            model=model_name,
            device=0 if device == "cuda" else -1,
            pad_token_id=50256,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        return np.array([0.0, 0.0, 0.0], dtype=np.float32), {}

    def step(self, action):
        self.current_step += 1

        scalar = float(np.clip(action[0], 0.0, 1.0))
        idx = int(scalar * (self.n_prompts - 1))
        example = self.dataset[idx]
        prompt = example["query"]
        is_poisoned = bool(example["is_poisoned"])

        try:
            output = self.generator(prompt, max_new_tokens=30, do_sample=False)
            generated = output[0]["generated_text"][len(prompt):]
        except Exception:
            generated = ""

        sentiment = TextBlob(generated).sentiment.polarity

        if is_poisoned:
            reward = 1.0 if sentiment < 0 else -1.0
        else:
            reward = 0.0

        terminated = self.current_step >= self.max_steps
        obs = np.array(
            [sentiment, float(is_poisoned), self.current_step / self.max_steps],
            dtype=np.float32,
        )
        return obs, reward, terminated, False, {"sentiment": sentiment, "is_poisoned": is_poisoned}
