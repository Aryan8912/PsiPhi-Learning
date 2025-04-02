import copy
import functools as ft
from typing import NamedTuple, Optional, Sequence, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
import ml_collections
import optax
import rlax

from social_rl import losses
from social_rl import parts
from social_rl import tree_utils
from social_rl.agents.itd import ITDNetworkOutput
from social_rl.networks import GridworldConvEncoder, LayerNormMLP

class PsiPhiLearnerState(NamedTuple):
    target_params: hk.Params
    opt_state: optax.OptState
    num_unique_steps: int

class PsiPhiActorState(NamedTuple):
    network_state: parts.State
    num_unique_steps.State
    preference_vector: Optional[chex.Array] = None

def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()
    config.num_cumulants = 7
    config.num_demonstrators = 2
    config.learning_rate = 1e-4
    config.learning_rate = 1e-4
    config.gamma = 0.9
    config.min_actor_steps_before_learning = 1_000
    config.train_every = 1
    config.train_epsilon = ml_collections.ConfigDict()
    config.train_epsilon.init_value = 1.0
    config.train_epsilon.end_value = 0.05
    config.train_epsilon.transition_steps = 50_000
    config.eval_epsilon = 0.00
    config.update_target_every = 1_000
    return config

@chex.dataclass(frozen=True)
class PsiPhiNetworkOutput:
    cumulants: chex.Array
    others_successor_features: chex.Array
    ego_successor_features: chex.Array
    others_preference_vectors: chex.Array
    ego_preference_vector: chex.Array

    others_policy_params: chex.Array
    others_rewards: chex.Array
    ego_action_value: chex: Array

class PsiPhiNetwork(nk.Module):

    def __init__(
            self, 
            num_actions: int,
            
    )