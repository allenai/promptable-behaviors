from typing import Dict, Optional, Callable, cast, Tuple, Union
import torch

from allenact.algorithms.onpolicy_sync.losses.abstract_loss import (
    AbstractActorCriticLoss,
    ObservationType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput


class KLLoss(AbstractActorCriticLoss):
    def __init__(
            self,
            alpha: float = 1.0,
            codebook_probs_uuid: str = 'codebook_probs',
            mean: float = 0.0,
            std: float = 1.0
    ):
        self.alpha = alpha
        self.codebook_probs_uuid = codebook_probs_uuid
        self.mean = mean
        self.std = std

    def loss(  # type: ignore
            self,
            step_count: int,
            batch: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
            actor_critic_output: ActorCriticOutput[CategoricalDistr],
            *args,
            **kwargs
    ):
        loss = self.temporal_loss(step_count,
                                  batch,
                                  actor_critic_output,
                                  *args,
                                  **kwargs)
        # multiply weight
        loss *= self.alpha

        return (
            loss,
            {"kl_loss": loss.item() / self.alpha, },
        )

    def temporal_loss(  # type: ignore
            self,
            step_count: int,
            batch: Dict[str, Union[Dict[str, torch.Tensor], torch.Tensor]],
            actor_critic_output: ActorCriticOutput[CategoricalDistr],
            *args,
            **kwargs
    ):
        codebook_probs = actor_critic_output.extras[self.codebook_probs_uuid]
        normal_dist = torch.distributions.normal.Normal(self.mean, self.std)
        log_probs = normal_dist.log_prob(codebook_probs)
        kl_loss = -torch.mean(log_probs)

        return kl_loss
