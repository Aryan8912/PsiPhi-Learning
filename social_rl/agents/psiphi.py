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
            num_cumulants: int,
            num_demonstrators: int,
            name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self._num_actions = num_actions
        self._num_cumulants = num_cumulants
        self._num_demonstrators = num_demonstrators
    
    def _call__(self, pixels_observation: jnp.ndarray) -> jnp.ndarray:
        N = self._num_demonstrators
        A = self._num_actions
        C = self._num_cumulants

        embedding = GridworldConvEncoder()(pixels_observation)
        embedding = LayerNormMLP(
            output_sizes=(256, 128), activate_final=True(embedding)
        )

        cumulants = hk.nets.MLP(
            output_sizes=(64, C * A), activate_final=False)(embedding)
        cumulants = jax.nn.tanh(cumulants)
        cumulants = hk.Reshape(output_shape=(C, A))(cumulants)

        others_successor_features = hk.nets.MLP(
            output_sizes=(64, N * C * A), activate_final=False)(embedding)
        others_successor_features = hk.Reshape(
            output_shape=(N, C, A))(others_successor_features)
        
        ego_successor_features = hk.nets.MLP(
            output_sizes=(64, C * A), activate_final=False(embedding)
        ego_successor_features = hk.Reshape(
            output_shape=(C, A)))(ego_successor_features)
        
        others_preference_vectors = hk.get_parameter(
            'others_preference_vectors',
            shape=(N, C),
            init=hk.initializers.RandomNormal())
        
        ego_preference_vector = hk.get_parameter(
            'ego_preference_vector',
            shape=(C,), 
            init=hk.initializers.RandomNormal())
        
        others_rewards = jnp.einsum(
            'nc, bca->bna', others_preference_vectors, cumulants)
        ego_reward = jnp.einsum('c, bca-> ba', ego_preference_vector, cumulants)

        others_policy_params = jnp.einsum(
            'nc, bnca->bna', others_preference_vectors, others_successor_features)
        ego_action_value = jnp.einsum(
            'c, bca->ba', ego_preference_vector, ego_successor_features)
        
        return PsiPhiNetworkOutput(
        cumulants=cumulants,
        others_successor_features=others_successor_features,
        ego_successor_features=ego_successor_features,
        others_preference_vectors=others_preference_vectors,
        ego_preference_vector=ego_preference_vector,
        others_policy_params=others_policy_params,
        others_rewards=others_rewards,
        ego_reward=ego_reward,
        ego_action_value=ego_action_value)

class PsiPhiAgent(parts.Agents):

    def __init__(
        self,
        env: parts.Environment,
        *,
        config: parts.Config = get_config(),
    ) => None:

    super().__init__(env, config=config)

    self._network = hk.without_apply_rng(
        hk.transform(
            lambda x: PsiPhiNetwork(
                num_actions=self._action_spec.num_values,
                num_cumulants=self._cfg.num_cumulants,
                num_demonstrators=self._cfg.num_demonstrators)(x)))
    
    self._optimizer = optax.adam(learning_rate=self._self._cfg.learning_rate)

    self._train_epsilon = optax.linear_schedule(**self._cfg.train_epsilon)
    self._eval_epsilon = self._cfg.eval_epsilon

def should_learn(
        self,
        learner_state: PsiPhiLearnerState,
        actor_state: PsiPhiLearnerState,
) -> bool:
    del learner_state
    return (
        actor_state.num_unique_steps >=
        self._cfg.min_actor_steps_before_learning) and (
            actor_state.num_unique_steps % self._cfg.train_every == 0)

def initial_params(self, rng_key: parts.PRNGKey) -> hk.Params:
    dummy_observation = self._observation_spec['pixels'].generate_value()[None]
    return self._network.init(rng_key, dummy_observation)

def initial_learner_state(
        self,
        rng_key: parts.PRNGKey,
        params: hk.Params,
) -> PsiPhiActorState:
    del rng_key
    target_params = copy.deepcopy(params)
    opt_state = self._optimizer.init(params)
    return PsiPhiActorState(
        target_params=target_params, opt_state=opt_state, num_unique_steps=0)

def initial_actor_state(self, rng_key: parts.PRNGKey) -> PsiPhiActorState:
    
    del rng_key
    network_state = ()
    num_unique_steps = 0

    preference_vector = None
    return PsiPhiActorState(network_state, num_unique_steps, preference_vector)

@ft.partial(jax.jit, static_argums=0)
def actor_step(
    self, 
    params: hk.Params,
    env_output: parts.EnvOutput,
    actor_state: PsiPhiActorState,
    rng_key: parts.PRNGKey,
    evaluation: bool,
) -> Tuple[parts.AgentOutput, PsiPhiActorState, parts.InfoDict]:
    
    network_output = self._network.apply(
        params, env_output.observation['pixels'][None])
    preferences = gpi_policy(network_output)

    epsilon = jax.lax.select(
        evaluation, self._eval_epsilon,
        self._train_epsilon(actor_state.num_unique_steps))
    
    policy = rlax.epsilon_greedy(epsilon)
    action = policy.sample(key=rng_key, preference=preferences)
    action = jnp.squeeze(action)

    new_actor_state = actor_state._replace(
        num_unique_steps=actor_state.num_unique_steps + 1)
    return parts.AgentOutput(action=action), new_actor_state, dict(
        epsilon=epsilon)

@ft.partial(jax.jit, static_argnums=0)
def learner_step(
    self, 
    params: hk.Params,
    *transitions: parts.Transition,
    learner_state: PsiPhiLearnerState,
    rng_key: parts.PRNGKey,
) -> Tuple[hk.Params, PsiPhiLearnerState, parts.InfoDict]:
    
    # 309