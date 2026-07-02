from torch import nn
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.policies import ActorCriticPolicy

# TODO: Create CNN and RNN versions of the policy

class MoActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, n_objectives, **kwargs):
        self.n_objectives = n_objectives
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)

    def _build(self, lr_schedule: Schedule) -> None:
        super()._build(lr_schedule)
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