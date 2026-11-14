"""asymmetric_policy.py -- MlpPolicy with a privileged critic (D-11, D-15).

Stable-Baselines3 feeds the *same* observation vector to the actor and the
critic.  The D-2..D-58 design appends a privileged block to the end of the
observation vector::

    [actor frames | future frames | privileged fields]

where ``privileged fields`` are critic-only (``time_remaining``,
``facing_error``, ``target_true_pos``, ``target_true_vel`` -- D-12, D-15).
The actor must not see them.

SB3 2.x ``ActorCriticPolicy.forward()`` (share_features_extractor=True path)::

    features = self.extract_features(obs)          # → single Tensor, full obs
    latent_pi, latent_vf = self.mlp_extractor(features)

So our strategy is:
  1. Do NOT override ``extract_features`` -- let SB3 return the full obs tensor.
  2. Override ``_build_mlp_extractor`` to create a ``_SplitExtractor`` that
     internally slices ``features[:, :actor_obs_dim]`` for the actor branch
     and uses the full ``features`` for the critic branch.
  3. Expose ``latent_dim_pi`` / ``latent_dim_vf`` so SB3's ``_build()`` can
     size the action/value heads.

``training/algos.py::build_model(asymmetric=True)`` passes ``actor_obs_dim``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy, MlpExtractor


__all__ = ["AsymmetricActorCriticPolicy"]


class _SplitExtractor(nn.Module):
    """Two independent ``MlpExtractor`` branches behind a single module.

    SB3 (share_features_extractor=True path) calls::

        latent_pi, latent_vf = mlp_extractor(features)

    where ``features`` is the *full* observation tensor (shape: [B, obs_dim]).
    This extractor:
      - feeds ``features[:, :actor_obs_dim]`` to the *actor* branch
      - feeds the full ``features``                to the *critic* branch

    so the critic sees the privileged columns while the actor does not.
    """

    def __init__(
        self,
        actor_extractor: MlpExtractor,
        critic_extractor: MlpExtractor,
        actor_obs_dim: int,
    ):
        super().__init__()
        self.actor = actor_extractor
        self.critic = critic_extractor
        self._actor_obs_dim = int(actor_obs_dim)

    # SB3 reads these in ActorCriticPolicy._build() to size the action/value heads.
    @property
    def latent_dim_pi(self) -> int:
        return self.actor.latent_dim_pi

    @property
    def latent_dim_vf(self) -> int:
        return self.critic.latent_dim_vf

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """Actor branch -- receives full obs, slices to actor width."""
        return self.actor.forward_actor(features[:, : self._actor_obs_dim])

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """Critic branch -- receives full obs including privileged columns."""
        return self.critic.forward_critic(features)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """SB3 calls this as ``latent_pi, latent_vf = mlp_extractor(features)``."""
        return self.forward_actor(features), self.forward_critic(features)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """``MlpPolicy`` whose critic may receive extra (privileged) columns.

    Parameters
    ----------
    actor_obs_dim:
        Width of the *actor* slice (everything except the privileged block).
        Supplied via ``policy_kwargs``; recorded in ``_get_constructor_parameters``
        so ``save()``/``load()`` round-trips are lossless (D-15).
    """

    def __init__(self, *args, actor_obs_dim: Optional[int] = None, **kwargs):
        if actor_obs_dim is None:
            raise ValueError(
                "AsymmetricActorCriticPolicy requires 'actor_obs_dim' in "
                "policy_kwargs (training/algos.py build_model passes it) so the "
                "value is recorded in the saved checkpoint."
            )
        self.actor_obs_dim = int(actor_obs_dim)
        if self.actor_obs_dim <= 0:
            raise ValueError("actor_obs_dim must be > 0")
        super().__init__(*args, **kwargs)
        if self.actor_obs_dim > self.features_dim:
            raise ValueError(
                f"actor_obs_dim ({self.actor_obs_dim}) exceeds the observation "
                f"width ({self.features_dim})"
            )
        if self.actor_obs_dim == self.features_dim:
            raise ValueError(
                "actor_obs_dim equals the observation width: use MlpPolicy, "
                "there are no privileged columns."
            )

    # ------------------------------------------------------------------ SB3

    def _get_constructor_parameters(self) -> dict:
        """Re-declare ``actor_obs_dim`` so ``load()`` can rebuild the policy."""
        params = super()._get_constructor_parameters()
        params["actor_obs_dim"] = self.actor_obs_dim
        return params

    def _build_mlp_extractor(self) -> None:
        """Build two independent extractors -- one per branch input width.

        SB3 calls this inside ``_build()``.  After it returns, ``_build()``
        reads ``self.mlp_extractor.latent_dim_pi`` and
        ``self.mlp_extractor.latent_dim_vf`` to size the action/value heads,
        so both properties must be present on the returned object.
        """
        pi_arch: List[int] = list(self.net_arch.get("pi", [])) or [64, 64]
        vf_arch: List[int] = list(self.net_arch.get("vf", [])) or [64, 64]

        actor_extractor = MlpExtractor(
            self.actor_obs_dim, pi_arch, activation_fn=self.activation_fn
        )
        critic_extractor = MlpExtractor(
            self.features_dim, vf_arch, activation_fn=self.activation_fn
        )
        self.mlp_extractor = _SplitExtractor(
            actor_extractor, critic_extractor, self.actor_obs_dim
        )

    # extract_features is intentionally NOT overridden.
    # SB3's default returns the full obs as a flat Tensor, which is exactly
    # what _SplitExtractor.forward() expects (it slices internally).
