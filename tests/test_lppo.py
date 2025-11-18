from stable_baselines3 import PPO
from stable_baselines3.common.buffers import MoRolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import MoVecEnv, MoDummyVecEnv, DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.lppo import LPPO, seqLPPO

import numpy as np
import gymnasium as gym

class UnbreakableBottlesEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    # Actions: 0=left, 1=right, 2=pick_up_bottle
    def __init__(self, max_steps=50, mode="vector", we=2):
        super().__init__()
        self.mode = mode
        self.we = we
        self.num_cells = 5
        self.agent_start = 0
        self.agent_goal = 4
        self.max_bottles = 2
        self.drop_prob = 0.1
        self.max_steps = max_steps
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self._rng = np.random.default_rng()
        self.reset()

    def reset(self, *, seed=None, options=None):

        self.agent_loc = self.agent_start
        self.bottles_carried = 0
        self.bottles_delivered = 0
        self.num_bottles = [0 for _ in range(self.num_cells - 2)]
        self.steps = 0
        obs = self._obs()
        return obs, {}

    def step(self, action):
        self.steps += 1
        truncated = self.steps >= self.max_steps
        info = {}
        bottles_delivered_this_step = 0
        before_floor_bottles = min(sum(self.num_bottles), 1)

        # Left
        if action == 0:
            if self.agent_loc > 0:
                self.agent_loc -= 1
            if self.agent_loc > 0 and self.bottles_carried == self.max_bottles:
                if self._rng.uniform() < self.drop_prob:
                    self.num_bottles[self.agent_loc - 1] += 1
                    self.bottles_carried -= 1

        # Right
        elif action == 1:
            if self.agent_loc < self.agent_goal:
                self.agent_loc += 1
            if self.agent_loc == self.agent_goal and self.bottles_carried > 0:
                bottles_delivered_this_step = min(self.max_bottles, self.bottles_carried)
                self.bottles_delivered += bottles_delivered_this_step
                self.bottles_carried -= bottles_delivered_this_step
            elif self.agent_loc < self.agent_goal and self.bottles_carried == self.max_bottles:
                if self._rng.uniform() < self.drop_prob:
                    self.num_bottles[self.agent_loc - 1] += 1
                    self.bottles_carried -= 1

        # Pick up bottle
        elif action == 2:
            if self.agent_loc == self.agent_start and self.bottles_carried < self.max_bottles:
                self.bottles_carried += 1
            elif self.agent_loc < self.agent_goal and self.bottles_carried < self.max_bottles:
                idx = self.agent_loc - 1
                if self.num_bottles[idx] > 0:
                    self.num_bottles[idx] -= 1
                    self.bottles_carried += 1

        after_floor_bottles = min(sum(self.num_bottles), 1)
        terminated = self.bottles_delivered >= self.max_bottles

        r0 = before_floor_bottles - after_floor_bottles
        r1 = -1 + bottles_delivered_this_step * 25
        if self.mode == "vector":
            reward = np.array([r0, r1])
        else:
            reward = r0 * self.we + r1
        info["r0"] = r0
        info["r1"] = r1
        obs = self._obs()
        return obs, reward, terminated, truncated, info

    def _obs(self):
        # Minimal encoding: [agent_loc, bottles_carried, bottles_delivered, sum(num_bottles)]
        obs = np.array([self.agent_loc, self.bottles_carried, self.bottles_delivered,
                        self.num_bottles[0] * 2 ** 0 + self.num_bottles[1] * 2 ** 1 + self.num_bottles[2] * 2 ** 2],
                       dtype=np.float32)

        return obs / np.array([5 - 1, 3 - 1, 2 - 1, 2 ** 3 - 1])  # Normalize for RL input

    def render(self):
        if self.steps == 0:
            print("|\tS\t|\t1\t|\t2\t|\t3\t|\tD\t|")
        label_agent_loc = str(self.agent_loc)
        label_agent_loc = "S" if label_agent_loc == "0" else label_agent_loc
        label_agent_loc = "D" if label_agent_loc == "4" else label_agent_loc
        print(
            f'Agent:{label_agent_loc} Carry:{self.bottles_carried} Delivered:{self.bottles_delivered} Floor:{self.num_bottles}')


def linear_schedule(initial_value):
    def func(progress_remaining):
        return progress_remaining * initial_value

    return func

def linear_then_flat(initial_value: float, final_value: float = 5e-6, elbow: float = 0.25):
    def func(progress_remaining: float) -> float:
        finish = 1.0 - elbow  # where decay ends and flat begins
        if progress_remaining > finish:
            # Linear decay phase
            linear_progress = (progress_remaining - finish) / elbow
            return final_value + (initial_value - final_value) * linear_progress
        else:
            # Flat phase
            return final_value

    return func


from stable_baselines3.common.monitor import MoMonitor
from stable_baselines3.common.vec_env import MoVecMonitor


class EntropyAnnealingCallback(BaseCallback):
    """
    Callback for annealing entropy coefficient over training.

    :param start_ent_coef: Initial entropy coefficient.
    :param end_ent_coef: Minimum entropy coefficient after annealing.
    :param total_timesteps: Number of timesteps over which to anneal entropy.
    :param schedule: "linear" or "exponential" (default: "exponential").
    :param verbose: Verbosity level (0: no output, 1: some output).
    """

    def __init__(self, start_ent_coef: float, end_ent_coef: float, total_timesteps: int,
                 schedule: str = "linear", verbose: int = 0):
        super().__init__(verbose)
        self.start_ent_coef = start_ent_coef
        self.end_ent_coef = end_ent_coef + 1e-08
        self.total_timesteps = total_timesteps
        self.schedule = schedule.lower()
        self.ent_coef = start_ent_coef

    def _on_training_start(self) -> None:
        """
        Check if the policy supports entropy coefficient adjustment.
        """
        assert hasattr(self.model, "ent_coef"), "This model does not support entropy coefficient annealing."
        self.model.ent_coef = self.start_ent_coef  # Set initial value

    def _on_step(self) -> bool:
        """
        Update entropy coefficient at each step.
        """
        progress = min(1.0, self.num_timesteps / int(self.total_timesteps*0.65))  # Ensure max value is 1.0. We only decay for 80% of the training time, then keep it constant
        if progress >= 1.0:
            return True
        if self.schedule == "linear":
            self.ent_coef = self.start_ent_coef + progress * (self.end_ent_coef - self.start_ent_coef)
        else:  # Exponential decay
            decay_rate = np.log(self.end_ent_coef / self.start_ent_coef) / self.total_timesteps
            self.ent_coef = self.start_ent_coef * np.exp(decay_rate * self.num_timesteps)

        self.ent_coef = max(self.end_ent_coef, self.ent_coef)  # Ensure it doesn't go below minimum
        self.model.ent_coef = self.ent_coef  # Update model entropy coefficient

        if self.verbose > 0 and self.n_calls % 1000 == 0:  # Log every 1000 steps
            print(f"Step {self.num_timesteps}: Entropy Coef = {self.ent_coef:.6f}")

        return True  # Continue training




def make_env():
    def _init():
        return UnbreakableBottlesEnv(mode="vector")
    return _init
env = MoDummyVecEnv([make_env() for _ in range(8)],)
env = MoVecMonitor(env)


def linear_schedule_then_flat(initial_value):
    def func(progress_remaining):
        return max(progress_remaining * initial_value, 1)  # Linearly decrease

    return func

args = {
    "n_steps": 512,
    "batch_size": 256,
    "ent_coef": 0.06,
    "learning_rate": [linear_schedule(8e-4), linear_schedule(8e-3)],
    "rollout_buffer_class": MoRolloutBuffer,
    "rollout_buffer_kwargs": {},
    "beta_values": [3.0, 1.0],
    "eta_values": 1e-3*2,
    "policy_kwargs": {
        "n_objectives": 2,
        "net_arch": dict(pi=[128, 128], vf=[128, 128])
    },
    "verbose": 1,
    "device": "cuda:0",
    "tensorboard_log": "runs",
    "clip_range_vf": 0.2,
    "gamma": 1.0,
    "normalize_advantage": True,
    "tolerance": linear_then_flat(0.5, 0.05, 0.5),
    "recent_loses_len": 128,
    "n_epochs": 40
}

model = seqLPPO("MoMlpPolicy", env, 2, **args)


#model = PPO("MlpPolicy", env, **args)

model.learn(total_timesteps=2000000, log_interval=1, callback=[
    EntropyAnnealingCallback(args["ent_coef"], 0.0001, 1000000)
])
model.save("test_model")
"""
env = UnbreakableBottlesEnv()
model = PPO.load("test_model", env=env)

expected_mo_return = np.zeros((1000, 2))
for ep in range(10):
    obs, _ = env.reset()
    for i in range(env.max_steps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, tr, tm, info = env.step(action.item())
        env.render()
        print(reward)
        expected_mo_return[ep] += reward
        if tr or tm:
            break

print(expected_mo_return.mean(axis=0))"""