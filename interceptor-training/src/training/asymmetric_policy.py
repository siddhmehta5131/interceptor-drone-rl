"""asymmetric_policy.py -- MlpPolicy with a privileged critic (D-11, D-15).

Stable-Baselines3 feeds the *same* observation vector to the actor and the
critic.  The D-2..D-58 design appends a privileged block to the end of the
observation vector:

    ``[actor frames | future frames | privileged fields]``

where ``privileged fields`` are critic-only (``time_remaining``,
``facing_error``, ``target_true_pos``, ``target_true_vel`` -- D-12, D-15).
The actor must not see them, otherwise the critic loses its advantage-based
signal and the policy is trained on information it cannot measure on the real
vehicle.

SB3 has no native asymmetric support, so this module provides:

``AsymmetricActorCriticPolicy``
    ``MlpPolicy`` whose ``extract_features`` slices the observation at
    ``actor_obs_dim`` and builds **two independent MLP extractors** (one for
    the actor on the sliced observation, one for the critic on the full
    observation).  Two extractors are required because the input widths
    differ -- a single shared extractor cannot consume both.

``policy_kwargs_for(model_def, obs)``
    Chooses the right ``policy_class`` + ``policy_kwargs`` for a model
    definition: the asymmetric policy when the model declares privileged
    critic fields, plain ``MlpPolicy`` otherwise.

``actor_obs_dim``
    Supplied through ``policy_kwargs`` (never through a module-level global)
    **and** re-declared by ``_get_constructor_parameters()`` so the value
    survives ``model.save()`` / ``PPO.load()`` round-trips.  SB3 folds the
    returned dict into ``data["policy_kwargs"]`` when it builds the
    constructor arguments on load, which is the documented extension point
    for policies with non-standard constructor arguments.

Notes
-----
* ``lr_schedule`` is deliberately *not* forwarded to ``super().__init__``:
  SB3 >= 2.0 removed it from the signature (it is now a class attribute set
  by the learning-rate schedule).  Passing it raises ``TypeError`` on SB3
  >= 2.4, which is the floor declared in ``requirements.txt``.
* This module imports torch/SB3 at module scope and therefore may only be
  imported when they are installed.  ``src.training.__init__`` deliberately
  does *not* import it so the host-side smoke test keeps working on a plain
  Python install.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch as th
from torch import nn

from stable_baselines3.common.policies import ActorCriticPolicy, MlpExtractor


__all__ = [
    "AsymmetricActorCriticPolicy",
    "policy_kwargs_for",
]


class _SplitExtractor(nn.Module):
    """Holds one :class:`MlpExtractor` per branch behind one module.

    SB3's ``ActorCriticPolicy`` expects ``mlp_extractor`` to expose
    ``forward_actor(latent)`` and ``forward_critic(latent)``.  We give it two
    independent extractors so each branch sees its own input width.
    """

    def __init__(self, actor_extractor: MlpExtractor, critic_extractor: MlpExtractor):
        super().__init__()
        self.actor = actor_extractor
        self.critic = critic_extractor

    def forward_actor(self, latent_pi: th.Tensor) -> th.Tensor:
        return self.actor.forward_actor(latent_pi)

    def forward_critic(self, latent_pi: th.Tensor) -> th.Tensor:
        return self.critic.forward_critic(latent_pi)

    def forward(self, latent_pi: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.forward_actor(latent_pi), self.forward_critic(latent_pi)


class AsymmetricActorCriticPolicy(ActorCriticPolicy):
    """``MlpPolicy`` whose critic may receive extra (privileged) columns.

    Parameters
    ----------
    actor_obs_dim:
        Width of the *actor* slice, i.e. everything except the privileged
        block.  Supplied via ``policy_kwargs``.

    The full observation width (``features_dim``) is the critic width.
    """

    def __init__(self, *args, actor_obs_dim: Optional[int] = None, **kwargs):
        if actor_obs_dim is None:
            raise ValueError(
                "AsymmetricActorCriticPolicy requires 'actor_obs_dim' in "
                "policy_kwargs; build it with policy_kwargs_for() so the "
                "value is recorded in the saved checkpoint"
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
                "there are no privileged columns"
            )

    # ------------------------------------------------------------------ SB3

    def _get_constructor_parameters(self) -> dict:
        """Re-declare ``actor_obs_dim`` so ``load()`` can rebuild the policy.

        SB3 serialises ``policy_kwargs`` verbatim; a custom constructor
        argument that is not present in the original ``policy_kwargs`` would
        therefore be lost.  Adding it here keeps ``save``/``load`` symmetric
        (D-15) even when a caller constructs the policy by hand.
        """
        params = super()._get_constructor_parameters()
        params["actor_obs_dim"] = self.actor_obs_dim
        return params

    def _build_mlp_extractor(self) -> None:
        """Build *two* extractors: one per branch input width.

        ``net_arch`` is the shared ``{pi: [...], vf: [...]}`` mapping.  The
        actor extractor consumes the *sliced* observation that
        ``extract_features`` returns as ``features_pi`` (width
        ``actor_obs_dim``); the critic extractor consumes the full
        observation (``features_vf``, width ``features_dim``).
        """
        pi_arch: List[int] = list(self.net_arch.get("pi", [])) or [64, 64]
        vf_arch: List[int] = list(self.net_arch.get("vf", [])) or [64, 64]

        actor_extractor = MlpExtractor(self.actor_obs_dim, pi_arch, activation_fn=self.activation_fn)
        critic_extractor = MlpExtractor(self.features_dim, vf_arch, activation_fn=self.activation_fn)
        self.mlp_extractor = _SplitExtractor(actor_extractor, critic_extractor)

    def extract_features(  # type: ignore[override]
        self, obs: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor]:
        """Split the observation into ``(actor_obs, critic_obs)``.

        SB3 calls this from ``forward()`` and expects the return value to be
        ``(features_pi, features_vf)``.  The actor branch therefore gets the
        narrow slice and the critic branch gets the full observation
        including the privileged columns.
        """
        if not th.is_floating_point(obs):
            obs = obs.float()
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return obs[:, : self.actor_obs_dim], obs


def policy_kwargs_for(
    model: object,
    obs_dim: int,
    *,
    policy_kwargs: Optional[dict] = None,
    privileged_fields: Optional[object] = None,
) -> Tuple[type, dict]:
    """Return ``(policy_class, policy_kwargs)`` for a model definition.

    ``model`` is a :class:`~src.utils.model_loader.ModelDef`; only two
    attributes are read (duck-typed on purpose so this module stays free of
    a hard config dependency).  ``privileged_fields`` overrides the model's
    own list -- used when a caller wants to train a symmetric policy on the
    same observation.

    Returns ``(MlpPolicy, kwargs)`` when there are no privileged columns, so
    the caller can pass the result straight to ``PPO(policy=...)``.
    """
    fields = privileged_fields if privileged_fields is not None else getattr(
        model, "privileged_critic", ()
    )
    fields = tuple(fields or ())
    kwargs = dict(policy_kwargs or {})

    if not fields:
        from stable_baselines3.common.policies import MlpPolicy

        return MlpPolicy, kwargs

    from ..envs.obs_builder import PRIVILEGED_FIELDS

    priv_dim = 0
    for name in fields:
        if name not in PRIVILEGED_FIELDS:
            raise ValueError(
                f"unknown privileged critic field {name!r}; "
                f"known fields: {sorted(PRIVILEGED_FIELDS)}"
            )
        priv_dim += PRIVILEGED_FIELDS[name]

    actor_dim = int(obs_dim) - priv_dim
    if actor_dim <= 0:
        raise ValueError(
            f"obs_dim={obs_dim} is not larger than the privileged block ({priv_dim})"
        )
    return AsymmetricActorCriticPolicy, {**kwargs, "actor_obs_dim": actor_dim}
