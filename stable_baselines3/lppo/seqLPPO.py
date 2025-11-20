from stable_baselines3.lppo.lppo import LPPO
import io
import pathlib
import sys
import time
from collections import deque
from typing import Union, Optional, Any, List

import numpy as np

import torch as th

from stable_baselines3.common.buffers import MoRolloutBuffer
from stable_baselines3.common.vec_env.patch_gym import _convert_space

from stable_baselines3.common.save_util import load_from_zip_file, recursive_setattr

from stable_baselines3.common.utils import get_system_info, check_for_correct_spaces, safe_mean, explained_variance

from stable_baselines3.common.type_aliases import GymEnv, Schedule

from stable_baselines3.common.base_class import SelfBaseAlgorithm
from torch.nn import functional as F

from gymnasium import spaces

from stable_baselines3.lppo.policies import MoActorCriticPolicy
from stable_baselines3.ppo.ppo import PPO
import warnings


class seqLPPO(LPPO):

    def __init__(self, policy, env, n_objectives, eta_values: List[Union[float, Schedule]] = None,
                 beta_values: List[float] = None,
                 tolerance: Union[float, Schedule] = 3e-5, recent_loses_len: int = 50, *args, **kwargs):
        super().__init__(policy, env, n_objectives, eta_values, beta_values, tolerance, recent_loses_len, *args,
                         **kwargs)

        self.active_obj = 0  # Goes from 0 to n_objectives-1

    def converged(self, minimum_updates=None) -> bool:
        """
        Check if the training has converged for the active objective.
        :return: (bool)
        """
        # If not enough updates have been performed, assume not converged
        minimum_updates = minimum_updates if minimum_updates is not None else self.recent_losses[0].maxlen
        if len(self.recent_losses[self.active_obj]) < minimum_updates:
            return False
        # Else check if the loss has stopped improving, within some tolerance
        else:
            tol = self.tolerance if isinstance(self.tolerance, float) else self.tolerance(
                self._current_progress_remaining)
            self.logger.record(f"train_mo/tolerance", tol)
            length = len(self.recent_losses[self.active_obj])
            # We compare the recent half of the buffer with the older half
            l_old_mean = -th.tensor(self.recent_losses[self.active_obj])[:int(length / 2)].mean().float()
            l_new_mean = -th.tensor(self.recent_losses[self.active_obj])[int(length / 2):].mean().float()
            # if diff is more than tol, we did not converge yet
            if abs(l_old_mean - l_new_mean) / abs(l_new_mean) > tol:
                return False
            else:
                return True

    def seq_update_lagrangian_multipliers(self):
        # We first gather the recent losses
        if self.converged():
            # We converged, we can move to the next objective
            print(f"Converged for reward function {self.active_obj}, t= {1 - self._current_progress_remaining}!")
            self.active_obj = 0 if self.active_obj == self.n_objectives - 1 else self.active_obj + 1
            self.recent_losses[self.active_obj].clear()

        else:
            length = int(self.recent_losses[self.active_obj].maxlen / 2)
            if len(self.recent_losses[self.active_obj]) < length:
                return

            for i in range(self.n_objectives - 1):
                if len(self.recent_losses[i]) > length + 1:
                    self.j[i] = -th.tensor(self.recent_losses[i])[length:].mean()
                    self.logger.record(f"train_mo/mean_recent_loss_{i}", self.j[i])

            # Then we update the lagrangian multipliers of the active objective and the ones before it.
            # note that if we are on the first objective, no update is performed
            for i in range(self.active_obj):
                eta = self.eta_values[i] if not callable(self.eta_values[i]) else self.eta_values[i](
                    self._current_progress_remaining)

                self.mu_values[i] += eta * (self.j[i] - (-self.recent_losses[i][-1]))
                self.mu_values[i] = max(self.mu_values[i], 0.0)

    def train(self):
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        if self.different_optimizers:
            self._update_learning_rate([self.policy.actor_optimizer, self.policy.critic_optimizer])
        else:
            self._update_learning_rate(self.policy.optimizer)

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, mo_losses = [], [], [[] for _ in range(self.n_objectives)]
        #pg_losses, value_losses, mo_losses = [], [], [[]]*self.n_objectives
        clip_fractions = []

        continue_training = True

        # Lexico update
        first_order_weights = self.get_scalarisation_weights()
        last_values = None
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            # Do a complete pass on the rollout buffer

            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.reshape(-1, self.n_objectives)

                # ENTROPY LOSS
                if entropy is None:
                    # Approximate entropy when no analytical form
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                # ACTOR LOSS
                # Normalize advantage
                advantages = rollout_data.advantages
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    # normalize advantages per objective
                    advantages = (advantages - advantages.mean(axis=0)) / (advantages.std(axis=0) + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                """scalarised_advantage = th.matmul(advantages[:, :self.active_obj + 1], first_order_weights[:self.active_obj + 1])
                scalarised_advantage = th.matmul(advantages, first_order_weights)
                if self.normalize_advantage and len(advantages) > 1:
                    scalarised_advantage = (scalarised_advantage - scalarised_advantage.mean(axis=0)) / (scalarised_advantage.std(axis=0) + 1e-8)"""

                # Compute losses separately

                unclipped_mo_actor_loss = th.column_stack([ratio for _ in range(self.n_objectives)]) * advantages
                clamped_ratio = th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                clamped_mo_actor_loss = th.column_stack([clamped_ratio for _ in range(self.n_objectives)]) * advantages
                mo_actor_loss = th.zeros(self.n_objectives, device=ratio.device)
                for j in range(self.n_objectives):
                    mo_actor_loss[j] = -th.min(unclipped_mo_actor_loss[:,j], clamped_mo_actor_loss[:,j]).mean()
                #mo_actor_loss = -th.min(unclipped_mo_actor_loss, clamped_mo_actor_loss,).mean(axis=0)
                no_entropy_mo_actor_loss = np.array(mo_actor_loss.detach().cpu())
                # Add entropy regularisation
                mo_actor_loss += self.ent_coef * entropy_loss

                actor_loss = th.matmul(mo_actor_loss[:self.active_obj + 1], first_order_weights[:self.active_obj + 1])

                # Compute losses from scalarised advantage instead of scalarised loss
                """unclipped_actor_loss = scalarised_advantage * ratio
                clamped_actor_loss =   scalarised_advantage * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                actor_loss = -th.min(unclipped_actor_loss, clamped_actor_loss).mean()
                actor_loss += self.ent_coef * entropy_loss"""

                for obj in range(self.n_objectives):
                    mo_losses[obj].append(no_entropy_mo_actor_loss[obj])

                # Logging
                pg_losses.append(np.dot(no_entropy_mo_actor_loss, first_order_weights.cpu()))
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                # CRITIC LOSS

                unclipped_loss = F.mse_loss(rollout_data.returns, values)

                if self.clip_range_vf is None:
                    # No clipping
                    # values_pred = values
                    value_loss = unclipped_loss
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling, normalisation recommended
                    if epoch > 0:
                        clipped_loss = F.mse_loss(
                            rollout_data.returns,
                            th.clamp(
                                values,
                                last_values - clip_range_vf,
                                last_values + clip_range_vf,
                            ))
                        value_loss = th.min(unclipped_loss, clipped_loss)
                    else:
                        # First iteration, no last_values
                        value_loss = unclipped_loss
                # Value loss using the TD(gae_lambda) target
                # value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())
                last_values = values.detach()

                entropy_losses.append(entropy_loss.item())

                # GENERAL LOSES
                actor_loss = th.matmul(mo_actor_loss[:self.active_obj + 1], first_order_weights[:self.active_obj + 1])

                critic_loss = self.vf_coef * value_loss

                global_loss = actor_loss + critic_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                if self.different_optimizers:
                    self.policy.actor_optimizer.zero_grad()
                    self.policy.critic_optimizer.zero_grad()

                    # seq training of actor.
                    actor_loss.backward()

                    critic_loss.backward()

                    th.nn.utils.clip_grad_norm_(self.policy.actor_params, self.max_grad_norm)
                    th.nn.utils.clip_grad_norm_(self.policy.critic_params, self.max_grad_norm)

                    self.policy.actor_optimizer.step()
                    self.policy.critic_optimizer.step()

                else:
                    # Optimization step
                    self.policy.optimizer.zero_grad()
                    global_loss.backward()
                    # Clip grad norm
                    th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

            for obj in range(self.n_objectives):
                self.recent_losses[obj].append(sum(mo_losses[obj]) / len(mo_losses[obj]))

            self.seq_update_lagrangian_multipliers()

            # logs
            self.logger.record("train/entropy_loss", np.mean(entropy_losses))
            self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
            self.logger.record("train/value_loss", np.mean(value_losses))

            self.logger.record("train/critic_loss", critic_loss.item())
            self.logger.record("train/actor_loss", actor_loss.item())

            self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
            self.logger.record("train/clip_fraction", np.mean(clip_fractions))
            self.logger.record("train/loss", global_loss.item())
            self.logger.record("train/ent_coef", self.ent_coef)

            # Lexico Logs
            for obj in range(self.n_objectives):
                self.logger.record(f"train_mo/advantage_scalarisation_weight_{obj}",
                                   first_order_weights[obj].float().item())
                if len(self.recent_losses[obj]) > 0:
                    self.logger.record(f"train_mo/loss_{obj}", np.mean(self.recent_losses[obj]))

            for obj in range(self.n_objectives - 1):
                self.logger.record(f"train_mo/mu_{obj}", self.mu_values[obj])
                if callable(self.eta_values[obj]):
                    self.logger.record(f"train_mo/eta_{obj}", self.eta_values[obj](self._current_progress_remaining))

            # self.logger.record("train/explained_variance", explained_var)
            if hasattr(self.policy, "log_std"):
                self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/clip_range", clip_range)
            if self.clip_range_vf is not None:
                self.logger.record("train/clip_range_vf", clip_range_vf)
