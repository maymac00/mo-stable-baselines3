from functools import partial

import numpy as np
import torch as th
from torch import nn

from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import MlpExtractor

# TODO: Create CNN and RNN versions of the policy


class VectorCritic(nn.Module):
    """One fully independent MLP critic per objective.

    Drop-in replacement for the single ``value_net`` head: takes the critic
    features and returns a ``(batch, n_objectives)`` value vector, but each
    objective's estimate comes from its own trunk, so gradients never mix
    across objectives.
    """

    def __init__(self, feature_dim, net_arch, activation_fn, n_objectives):
        super().__init__()
        critics = []
        for _ in range(n_objectives):
            layers = []
            last_dim = feature_dim
            for size in net_arch:
                layers.append(nn.Linear(last_dim, size))
                layers.append(activation_fn())
                last_dim = size
            layers.append(nn.Linear(last_dim, 1))
            critics.append(nn.Sequential(*layers))
        self.critics = nn.ModuleList(critics)

    def forward(self, features: th.Tensor) -> th.Tensor:
        return th.cat([critic(features) for critic in self.critics], dim=1)


class MoActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, n_objectives,
                 separate_critics: bool = False, **kwargs):
        self.n_objectives = n_objectives
        self.separate_critics = separate_critics
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _split_net_arch(self):
        if isinstance(self.net_arch, dict):
            return self.net_arch.get("pi", []), self.net_arch.get("vf", [])
        return self.net_arch, self.net_arch

    def _build_mlp_extractor(self) -> None:
        if not self.separate_critics:
            return super()._build_mlp_extractor()
        # The per-objective critics own their full trunks, so the shared
        # extractor keeps only the policy branch. Its value branch must not
        # exist at all: the dual-optimizer name filter ("value_net"/"vf")
        # would otherwise sweep those orphaned parameters into the critic
        # optimizer.
        pi_arch, _ = self._split_net_arch()
        self.mlp_extractor = MlpExtractor(
            self.features_dim,
            net_arch={"pi": pi_arch, "vf": []},
            activation_fn=self.activation_fn,
            device=self.device,
        )

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
        if self.separate_critics:
            # With an empty vf branch, latent_dim_vf == features_dim, so each
            # critic trunk runs from the extracted features up.
            _, vf_arch = self._split_net_arch()
            self.value_net = VectorCritic(
                self.mlp_extractor.latent_dim_vf, vf_arch, self.activation_fn, self.n_objectives
            )
            if self.ortho_init:
                for critic in self.value_net.critics:
                    critic.apply(partial(self.init_weights, gain=np.sqrt(2)))
                    output_layer = list(critic.children())[-1]
                    self.init_weights(output_layer, gain=1)
        else:
            self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, self.n_objectives)
            if self.ortho_init:
                self.init_weights(self.value_net, gain=1)
        # super()._build created the optimizer(s) *before* the value head was
        # swapped for the multi-objective one, so they tracked the orphaned
        # single-output head and the vector critic never trained. Rebuild them
        # over the current parameters.
        if self.different_optimisers:
            # Dual-optimizer mode (separate actor/critic learning rates).
            # ``lr_schedule`` is a length-2 list ``[actor_lr, critic_lr]``. Only
            # the critic optimizer needs rebuilding (it owns the swapped value
            # head); the actor params are unchanged.
            critic_params = [
                param
                for name, param in self.named_parameters()
                if "value_net" in name or "vf" in name
            ]
            self.critic_params = critic_params
            self.critic_optimizer = self.optimizer_class(
                critic_params, lr=lr_schedule[1](1), **self.optimizer_kwargs
            )
        else:
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1),
                                                  **self.optimizer_kwargs)

MoMlpPolicy = MoActorCriticPolicy
