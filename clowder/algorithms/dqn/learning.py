import clowder
from typing import Optional, Union, List, Dict
from clowder import specs
import reverb
from clowder import loggers
import tensorflow as tf
import torch

class DQNLearner(clowder.Learner, clowder.Saveable):
    def __init__(
        self, 
        network,
        target_network, 
        discount: float, 
        importance_sampling_exponent: float,
        learning_rate: float,
        target_update_period: int,
        dataset: tf.data.Dataset,
        max_abs_reward: Optional[float] = 1.,
        huber_loss_parameter: float = 1.,
        replay_client: Optional[reverb.Client] = None,
        logger: Optional[loggers.Logger] = None,
        checkpoint: bool = True,
        save_directory: str = '~/clowder',
        max_gradient_norm: Optional[float] = None,
    ):
        """Initializes the learner.

        Args:
        network: the online Q network (the one being optimized)
        target_network: the target Q critic (which lags behind the online net).
        discount: discount to use for TD updates.
        importance_sampling_exponent: power to which importance weights are raised
            before normalizing.
        learning_rate: learning rate for the q-network update.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        dataset: dataset to learn from, whether fixed or from a replay buffer (see
            `clowder.datasets.reverb.make_reverb_dataset` documentation).
        max_abs_reward: Optional maximum absolute value for the reward.
        huber_loss_parameter: Quadratic-linear boundary for Huber loss.
        replay_client: client to replay to allow for updating priorities.
        counter: Counter object for (potentially distributed) counting.
        logger: Logger object for writing logs to.
        checkpoint: boolean indicating whether to checkpoint the learner.
        save_directory: string indicating where the learner should save
            checkpoints and snapshots.
        max_gradient_norm: used for gradient clipping.
        """
        self._iterator = iter(dataset)
        self._network = network
        self._target_network = target_network
        #TODO: General Optimizer Module
        self._optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        self._replay_client = replay_client
        self._discount = discount
        self._target_update_period = target_update_period
        self._importance_sampling_exponent = importance_sampling_exponent
        self._max_abs_reward = max_abs_reward
        self._huber_loss_parameter = huber_loss_parameter
        if max_gradient_norm is None:
            max_gradient_norm = 1e10
        self._max_gradient_norm = max_gradient_norm
        
        # Learner State
        self._variables: List[List[specs.Tensor]] = [network.parameters()]
        self._num_steps = 0
        
        self._timestamp = None
    
    def _step(self) -> Dict[str, specs.Tensor]:
        inputs = next(self._iterator)
        transitions: specs.Transition = inputs.data
        keys, probs = inputs.info[:2]
        # Evaluate our networks.
        q_tm1 = self._network(transitions.observation)
        q_t_value = self._target_network(transitions.next_observation)
        q_t_selector = self._network(transitions.next_observation)
        
        r_t = transitions.reward.type(q_tm1.dtype)
        if self._max_abs_reward:
            r_t = torch.clamp(r_t, -self._max_abs_reward, self._max_abs_reward)
        d_t = transitions.discount.type(q_tm1.dtype) * self._discount.type(q_tm1.dtype)
        
    def step(self):
        # Do a batch of SGD
        
        