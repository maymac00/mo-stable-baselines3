import numpy as np

from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
from stable_baselines3.common.running_mean_std import RunningMeanStd
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from typing import Any, Optional, Union


class MoDummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns, n_objectives=2, *args, **kwargs):
        self.n_objectives = n_objectives
        super().__init__(env_fns, *args, **kwargs)
        self.buf_rews = np.zeros((self.num_envs, n_objectives), dtype=np.float32)


class MoVecEnv(SubprocVecEnv):
    def __init__(self, env_fns, n_objectives=2, *args, **kwargs):
        self.n_objectives = n_objectives
        super().__init__(env_fns, *args, **kwargs)
        self.buf_rews = np.zeros((self.num_envs, n_objectives), dtype=np.float32)


class MoVecNormalize(VecNormalize):
    """
    A subclass of VecNormalize that supports Vectorial (Multi-Objective) Rewards.

    :param venv: the vectorized environment to wrap
    :param reward_dim: The dimension of the reward vector. Default is 1 (scalar).
    :param kwargs: Arguments passed to the parent VecNormalize class.
    """

    def __init__(
            self,
            venv: MoVecEnv,
            **kwargs
    ):
        # Initialize the parent class
        super().__init__(venv, **kwargs)

        self.reward_dim = venv.n_objectives

        # If we have vectorial rewards, we must re-initialize the return RMS
        # to match the reward dimension (shape=(reward_dim,)).
        # The parent class initializes it as shape=() (scalar).
        if self.reward_dim > 1:
            self.ret_rms = RunningMeanStd(shape=(self.reward_dim,))

    def set_venv(self, venv: MoVecEnv) -> None:
        """
        Sets the vector environment and initializes the returns buffer
        with the correct dimension.
        """
        super().set_venv(venv)

        # Override self.returns initialization to handle vector shape
        if self.reward_dim > 1:
            self.returns = np.zeros((self.num_envs, self.reward_dim))

    def reset(self) -> Union[np.ndarray, dict[str, np.ndarray]]:
        """
        Reset all environments and the returns buffer.
        """
        # Call parent reset (which resets observations and sets returns to scalar zeros)
        obs = super().reset()

        # If multi-objective, we must re-zero self.returns with the correct vector shape
        # because super().reset() will have reset it to shape (num_envs,)
        if self.reward_dim > 1:
            self.returns = np.zeros((self.num_envs, self.reward_dim))

        return obs

    # We do not need to override step_wait, _update_reward, or normalize_reward.
    # The numpy operations in the parent class (broadcasting) work correctly
    # for vectors provided self.returns and self.ret_rms have the correct shapes
    # as defined in __init__ and set_venv.