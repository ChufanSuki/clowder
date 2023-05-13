import collections
import numpy as np
from clowder.ops import indexing_ops, base_ops
QExtra = collections.namedtuple("qlearning_extra", ["target", "td_error"])

def qlearning(q_tm1, a_tm1, r_t, pcont_t, q_t, name="QLearning"):
    """Implements the Q-learning loss as a op.

    The loss is `0.5` times the squared difference between `q_tm1[a_tm1]` and
    the target `r_t + pcont_t * max q_t`.

    See "Reinforcement Learning: An Introduction" by Sutton and Barto.
    (http://incompleteideas.net/book/ebook/node65.html).

    Args:
        q_tm1: Tensor holding Q-values for first timestep in a batch of
        transitions, shape `[B x num_actions]`.
        a_tm1: Tensor holding action indices, shape `[B]`.
        r_t: Tensor holding rewards, shape `[B]`.
        pcont_t: Tensor holding pcontinue values, shape `[B]`.
        q_t: Tensor holding Q-values for second timestep in a batch of
        transitions, shape `[B x num_actions]`.
        name: name to prefix ops created within this op.

    Returns:
        A namedtuple with fields:

        * `loss`: a tensor containing the batch of losses, shape `[B]`.
        * `extra`: a namedtuple with fields:
            * `target`: batch of target values for `q_tm1[a_tm1]`, shape `[B]`.
            * `td_error`: batch of temporal difference errors, shape `[B]`.
    """
    
    target = r_t + pcont_t * np.amax(q_t, axis=1)
    qa_tm1 = indexing_ops.batched_index(q_tm1, a_tm1)
    # Temporal difference error and loss.
    # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
    td_error = target - qa_tm1
    loss = 0.5 * np.square(td_error)
    return base_ops.LossOutput(loss, QExtra(target, td_error))