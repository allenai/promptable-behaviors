from collections import OrderedDict
from typing import Tuple, Dict, Optional, List, Sequence
from typing import TypeVar

import os
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

from gym.spaces.dict import Dict as SpaceDict
from omegaconf import DictConfig

from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearCriticHead,
    LinearActorHead,
    ObservationType,
    DistributionType,
)
from allenact.base_abstractions.distributions import CategoricalDistr
from allenact.base_abstractions.misc import ActorCriticOutput, Memory
from allenact.embodiedai.aux_losses.losses import MultiAuxTaskNegEntropyLoss
from allenact.embodiedai.models.aux_models import AuxiliaryModel
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from allenact.embodiedai.models.fusion_models import Fusion
from allenact.utils.model_utils import FeatureEmbedding
from allenact.utils.system import get_logger

# from procthor_visual_nav import cfg
from procthor_visual_nav.utils.sparsemax import Sparsemax

FusionType = TypeVar("FusionType", bound=Fusion)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, "float64")
        self.var = np.ones(shape, "float64")
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(
        mean, var, count, batch_mean, batch_var, batch_count
):
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count




class VisualNavActorCriticCodebook(ActorCriticModel[CategoricalDistr]):
    """Base class of visual navigation / manipulation (or broadly, embodied AI)
    model.

    `forward_encoder` function requires implementation.
    """

    action_space: gym.spaces.Discrete

    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            hidden_size=512,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[List[str]] = None,
            auxiliary_model_class=AuxiliaryModel,
            cfg: DictConfig = None,

    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.counter = 0
        self.cfg = cfg
        self.logging_path = f'{cfg.output_dir}/logs/{cfg.wandb.name}'
        self.pkl_path = f'{cfg.output_dir}/pkls/{cfg.wandb.name}'
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path)
        if not os.path.exists(self.pkl_path):
            os.makedirs(self.pkl_path)
        self._hidden_size = hidden_size
        assert multiple_beliefs == (beliefs_fusion is not None)
        self.multiple_beliefs = multiple_beliefs
        self.beliefs_fusion = beliefs_fusion
        self.auxiliary_uuids = auxiliary_uuids
        if isinstance(self.auxiliary_uuids, list) and len(self.auxiliary_uuids) == 0:
            self.auxiliary_uuids = None

        # Define the placeholders in init function
        self.state_encoders: Optional[nn.ModuleDict] = None
        self.aux_models: Optional[nn.ModuleDict] = None
        self.actor: Optional[LinearActorHead] = None
        self.critic: Optional[LinearCriticHead] = None
        self.prev_action_embedder: Optional[FeatureEmbedding] = None

        self.fusion_model: Optional[nn.Module] = None
        self.belief_names: Optional[Sequence[str]] = None
        self.auxiliary_model_class = auxiliary_model_class

        # codebook
        self.codebook_type = cfg.model.codebook.type
        self.codebook_indexing = cfg.model.codebook.indexing
        self.codebook_size = cfg.model.codebook.size
        self.code_dim = cfg.model.codebook.code_dim

        if self.cfg.model.codebook.initialization == "dictionary_learning":
            codebook_init = torch.tensor(np.load(f'{cfg.output_dir}/basis_vectors_baseline_5k_frames.npy'), dtype=torch.float32).contiguous()
            codebook_init_normalized = -1 + 2 * (codebook_init - codebook_init.min()) / (codebook_init.max() - codebook_init.min())
            self.codebook = torch.nn.Parameter(torch.Tensor(self.codebook_size, self.code_dim))
            self.codebook.data = codebook_init_normalized
        elif self.cfg.model.codebook.initialization == "random":
            self.codebook = torch.nn.Parameter(torch.randn(self.codebook_size, self.code_dim))
        self.codebook.requires_grad = True

        if self.codebook_indexing == "sparsemax":
            self.sparsemax = Sparsemax(dim=-1)

        # dropout to prevent codebook collapse
        self.dropout_prob = cfg.model.codebook.dropout
        self.dropout = nn.Dropout(self.dropout_prob)

        # codebook indexing
        if self.cfg.model.codebook.embeds == "joint_embeds":
            self.linear_codebook_indexer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1574, self.codebook_size),
            )
            self.linear_upsample = nn.Sequential(
                nn.Linear(self.code_dim, 1574),
            )
        elif self.cfg.model.codebook.embeds == "beliefs":
            self.linear_codebook_indexer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self._hidden_size, self.codebook_size),
            )
            self.linear_upsample = nn.Sequential(
                nn.Linear(self.code_dim, self._hidden_size),
            )



        # running mean and std
        self.rms = RunningMeanStd()

    def create_state_encoders(
            self,
            obs_embed_size: int,
            prev_action_embed_size: int,
            num_rnn_layers: int,
            rnn_type: str,
            add_prev_actions: bool,
            add_prev_action_null_token: bool,
            trainable_masked_hidden_state=False,
    ):
        rnn_input_size = obs_embed_size
        self.prev_action_embedder = FeatureEmbedding(
            input_size=int(add_prev_action_null_token) + self.action_space.n,
            output_size=prev_action_embed_size if add_prev_actions else 0,
        )
        if add_prev_actions:
            rnn_input_size += prev_action_embed_size

        state_encoders = OrderedDict()  # perserve insertion order in py3.6
        if self.multiple_beliefs:  # multiple belief model
            for aux_uuid in self.auxiliary_uuids:
                state_encoders[aux_uuid] = RNNStateEncoder(
                    rnn_input_size,
                    self._hidden_size,
                    num_layers=num_rnn_layers,
                    rnn_type=rnn_type,
                    trainable_masked_hidden_state=trainable_masked_hidden_state,
                )
            # create fusion model
            self.fusion_model = self.beliefs_fusion(
                hidden_size=self._hidden_size,
                obs_embed_size=obs_embed_size,
                num_tasks=len(self.auxiliary_uuids),
            )

        else:  # single belief model
            state_encoders["single_belief"] = RNNStateEncoder(
                rnn_input_size,
                self._hidden_size,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
            )

        self.state_encoders = nn.ModuleDict(state_encoders)

        self.belief_names = list(self.state_encoders.keys())

        get_logger().info(
            "there are {} belief models: {}".format(
                len(self.belief_names), self.belief_names
            )
        )

    def load_state_dict(self, state_dict, **kwargs):

        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "action_embedder" in key and "prev_action_embedder" not in key:
                continue
            elif "mem" in key:
                new_key = key.replace("mem", "codebook")
            elif "b_global_indexer" in key:
                new_key = key.replace("b_global_indexer", "linear_codebook_indexer")
            elif "b_global_projector" in key:
                new_key = key.replace("b_global_projector", "linear_upsample")
            elif "state_encoder." in key:  # old key name
                new_key = key.replace("state_encoder.", "state_encoders.single_belief.")
            elif "goal_visual_encoder.embed_class" in key:
                new_key = key.replace(
                    "goal_visual_encoder.embed_class", "goal_visual_encoder.embed_goal"
                )
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        return super().load_state_dict(new_state_dict, **kwargs)  # compatible in keys

    def create_actorcritic_head(self):
        self.actor = LinearActorHead(self._hidden_size, self.action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

    def create_aux_models(self, obs_embed_size: int, action_embed_size: int):
        if self.auxiliary_uuids is None:
            return
        aux_models = OrderedDict()
        for aux_uuid in self.auxiliary_uuids:
            aux_models[aux_uuid] = self.auxiliary_model_class(
                aux_uuid=aux_uuid,
                action_dim=self.action_space.n,
                obs_embed_dim=obs_embed_size,
                belief_dim=self._hidden_size,
                action_embed_size=action_embed_size,
            )

        self.aux_models = nn.ModuleDict(aux_models)

    @property
    def num_recurrent_layers(self):
        """Number of recurrent hidden layers."""
        return list(self.state_encoders.values())[0].num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        """The recurrent hidden state size of a single model."""
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
            memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
            for memory_key in self.belief_names
        }

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        raise NotImplementedError("Obs Encoder Not Implemented")

    def fuse_beliefs(
            self, beliefs_dict: Dict[str, torch.FloatTensor], obs_embeds: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        all_beliefs = torch.stack(list(beliefs_dict.values()), dim=-1)  # (T, N, H, k)

        if self.multiple_beliefs:  # call the fusion model
            return self.fusion_model(all_beliefs=all_beliefs, obs_embeds=obs_embeds)
        # single belief
        beliefs = all_beliefs.squeeze(-1)  # (T,N,H)
        return beliefs, None

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        obs_embeds = self.forward_encoder(observations)

        # 1.2 use embedding model to get prev_action embeddings
        if self.prev_action_embedder.input_size == self.action_space.n + 1:
            # In this case we have a unique embedding for the start of an episode
            prev_actions_embeds = self.prev_action_embedder(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions + 1,
                    other=torch.zeros_like(prev_actions),
                )
            )
        else:
            prev_actions_embeds = self.prev_action_embedder(prev_actions)
        joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)

        ###########################
        # Codebook before RNN
        ###########################
        if self.cfg.model.codebook.embeds == "joint_embeds":

            if self.codebook_indexing == "gumbel_softmax":
                codebook_probs = F.gumbel_softmax(self.linear_codebook_indexer(joint_embeds), tau=self.cfg.model.codebook.temperature, hard=True, dim=-1)
            elif self.codebook_indexing == "softmax":
                codebook_probs = F.softmax(self.linear_codebook_indexer(joint_embeds), dim=-1)
                if self.dropout_prob > 0:
                    codebook_probs = self.dropout(codebook_probs)
            elif self.codebook_indexing == "topk_softmax":
                softmax_output = F.softmax(self.linear_codebook_indexer(joint_embeds), dim=-1)
                if self.dropout_prob > 0:
                    softmax_output = self.dropout(softmax_output)
                topk_values, topk_indices = torch.topk(softmax_output, self.cfg.model.codebook.topk, dim=-1)
                codebook_probs = torch.zeros_like(softmax_output)
                codebook_probs.scatter_(-1, topk_indices, topk_values)
            elif self.codebook_indexing == "sparsemax":
                codebook_probs = self.sparsemax(self.linear_codebook_indexer(joint_embeds))
                if self.dropout_prob > 0:
                    codebook_probs = self.dropout(codebook_probs)



            self.rms.update(rearrange(codebook_probs, 'n b d -> (n b) d').detach().cpu().numpy())
            if self.codebook_type == "learned":
                code_output = torch.einsum('nbm,md->nbd', codebook_probs, self.codebook)
            elif self.codebook_type == "random":
                code_output = torch.einsum('nbm,md->nbd', codebook_probs, self.codebook.detach())
            elif self.codebook_type == "binary":
                code_output = torch.einsum('nbm,md->nbd', codebook_probs, 2.0 * (self.codebook > 0.) - 1)

            # code_output[:,:,5] *= 0.
            joint_embeds = self.linear_upsample(code_output)

        # 2. use RNNs to get single/multiple beliefs
        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            beliefs_dict[key], rnn_hidden_states = model(
                joint_embeds, memory.tensor(key), masks
            )
            memory.set_tensor(key, rnn_hidden_states)  # update memory here

        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(
            beliefs_dict, obs_embeds
        )  # fused beliefs

        ###########################
        # Codebook after RNN
        ###########################
        # if self.cfg.model.codebook.embeds == "beliefs":
        #     codebook_probs = F.softmax(self.linear_codebook_indexer(beliefs), dim=-1)
        #     codebook_probs = self.dropout(codebook_probs)
        #     self.rms.update(rearrange(codebook_probs, 'n b d -> (n b) d').detach().cpu().numpy())
        #     if self.codebook_type == "learned":
        #         code_output = torch.einsum('nbm,md->nbd', codebook_probs, self.codebook)
        #     elif self.codebook_type == "random":
        #         code_output = torch.einsum('nbm,md->nbd', codebook_probs, self.codebook.detach())
        #     elif self.codebook_type == "binary":
        #         code_output = torch.einsum('nbm,md->nbd', codebook_probs, 2.0 * (self.codebook > 0.) - 1)
        #     beliefs = self.linear_upsample(code_output)

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    "beliefs": (
                        beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs
                    ),
                    "obs_embeds": obs_embeds,
                    "aux_model": (
                        self.aux_models[aux_uuid]
                        if aux_uuid in self.aux_models
                        else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )
        if self.cfg.losses.KL_loss.alpha > 0:
            extras["codebook_probs"] = codebook_probs

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        distributions = self.actor(beliefs)
        values = self.critic(beliefs)
        actor_critic_output = ActorCriticOutput(
            distributions=distributions,
            values=values,
            extras=extras,
        )

        # logging
        if not self.cfg.eval and self.cfg.model.codebook.embeds == "joint_embeds" and self.counter > 0 and self.counter % self.cfg.logging.frequency == 0:
            torch.save(self.codebook.detach().cpu(), f'{self.logging_path}/codebook_{self.counter}.pt')
            torch.save(self.codebook.detach().cpu(), f'{self.logging_path}/codebook_latest.pt')
            np.save(f'{self.logging_path}/codebook_probs_{self.counter}.npy', self.rms.mean)
            np.save(f'{self.logging_path}/codebook_probs_latest.npy', self.rms.mean)

        # eval
        if self.cfg.eval:
            if self.counter == 0:
                torch.save(self.codebook.cpu(), f'{self.pkl_path}/codebook.pt')
            torch.save(codebook_probs.cpu(), f'{self.pkl_path}/codebook_probs_{self.counter}.pt')
            torch.save(code_output.cpu(), f'{self.pkl_path}/code_output_{self.counter}.pt')
            torch.save(joint_embeds.cpu(), f'{self.pkl_path}/joint_embeds_{self.counter}.pt')
            torch.save(prev_actions.cpu(), f'{self.pkl_path}/prev_actions_{self.counter}.pt')
            torch.save(beliefs.cpu(), f'{self.pkl_path}/beliefs_{self.counter}.pt')
            torch.save(distributions, f'{self.pkl_path}/distributions_{self.counter}.pt')
            torch.save(values.cpu(), f'{self.pkl_path}/values_{self.counter}.pt')
            torch.save(observations['goal_object_type_ind'].cpu(), f'{self.pkl_path}/goal_obj_{self.counter}.pt')
            torch.save(observations['rel_position_change']['dx_dz_dr'].cpu(),
                       f'{self.pkl_path}/dx_dz_dr_{self.counter}.pt')
            torch.save(observations['rel_position_change']['current_allocentric_position'].cpu(),
                       f'{self.pkl_path}/current_allocentric_position_{self.counter}.pt')
            torch.save(observations['visible_objects'].cpu(), f'{self.pkl_path}/visible_objects_{self.counter}.pt')
            torch.save(observations['visible_goal_objects'].cpu(), f'{self.pkl_path}/visible_goal_objects_{self.counter}.pt')
            torch.save(observations['total_reward'].cpu(), f'{self.pkl_path}/total_reward_{self.counter}.pt')
            torch.save(observations['distance_to_goal'].cpu(),
                       f'{self.pkl_path}/distance_to_goal_{self.counter}.pt')
            torch.save(observations['num_steps_taken'].cpu(), f'{self.pkl_path}/num_steps_taken_{self.counter}.pt')
            torch.save(observations['rgb'].cpu(), f'{self.pkl_path}/rgb_{self.counter}.pt')
            torch.save(observations['rgb_clip_resnet'].cpu(), f'{self.pkl_path}/rgb_clip_resnet_{self.counter}.pt')
            torch.save(observations['valid_moves_forward'].cpu(), f'{self.pkl_path}/valid_moves_forward_{self.counter}.pt')
            torch.save(observations['visited_pos'].cpu(), f'{self.pkl_path}/visited_pos_{self.counter}.pt')
            torch.save(observations['visited_room'].cpu(), f'{self.pkl_path}/visited_room_{self.counter}.pt')
            
            # torch.save(observations['floor_percentage'].cpu(), f'{self.pkl_path}/floor_percentage_{self.counter}.pt')

        self.counter += 1
        return actor_critic_output, memory


class VisualNavActorCritic(ActorCriticModel[CategoricalDistr]):
    """Base class of visual navigation / manipulation (or broadly, embodied AI)
    model.

    `forward_encoder` function requires implementation.
    """

    action_space: gym.spaces.Discrete

    def __init__(
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            hidden_size=512,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[List[str]] = None,
            cfg: DictConfig = None,
    ):
        super().__init__(action_space=action_space, observation_space=observation_space)
        self.counter = 0
        self.cfg = cfg
        self.pkl_path = f'{cfg.output_dir}/pkls/{cfg.wandb.name}'
        if not os.path.exists(self.pkl_path):
            os.makedirs(self.pkl_path)
        self._hidden_size = hidden_size
        assert multiple_beliefs == (beliefs_fusion is not None)
        self.multiple_beliefs = multiple_beliefs
        self.beliefs_fusion = beliefs_fusion
        self.auxiliary_uuids = auxiliary_uuids
        if isinstance(self.auxiliary_uuids, list) and len(self.auxiliary_uuids) == 0:
            self.auxiliary_uuids = None

        # Define the placeholders in init function
        self.state_encoders: Optional[nn.ModuleDict] = None
        self.aux_models: Optional[nn.ModuleDict] = None
        self.actor: Optional[LinearActorHead] = None
        self.critic: Optional[LinearCriticHead] = None
        self.prev_action_embedder: Optional[FeatureEmbedding] = None

        self.fusion_model: Optional[nn.Module] = None
        self.belief_names: Optional[Sequence[str]] = None

    def create_state_encoders(
            self,
            obs_embed_size: int,
            prev_action_embed_size: int,
            reward_info_embed_size: int,
            num_rnn_layers: int,
            rnn_type: str,
            add_prev_actions: bool,
            add_prev_action_null_token: bool,
            trainable_masked_hidden_state=False,
            num_objectives: int = 4,
    ):
        rnn_input_size = obs_embed_size
        self.prev_action_embedder = FeatureEmbedding(
            input_size=int(add_prev_action_null_token) + self.action_space.n,
            output_size=prev_action_embed_size if add_prev_actions else 0,
        )
        if add_prev_actions:
            rnn_input_size += prev_action_embed_size

        # add reward weights embeds size
        rnn_input_size += reward_info_embed_size*num_objectives

        state_encoders = OrderedDict()  # perserve insertion order in py3.6
        if self.multiple_beliefs:  # multiple belief model
            for aux_uuid in self.auxiliary_uuids:
                state_encoders[aux_uuid] = RNNStateEncoder(
                    rnn_input_size,
                    self._hidden_size,
                    num_layers=num_rnn_layers,
                    rnn_type=rnn_type,
                    trainable_masked_hidden_state=trainable_masked_hidden_state,
                )
            # create fusion model
            self.fusion_model = self.beliefs_fusion(
                hidden_size=self._hidden_size,
                obs_embed_size=obs_embed_size,
                num_tasks=len(self.auxiliary_uuids),
            )

        else:  # single belief model
            state_encoders["single_belief"] = RNNStateEncoder(
                rnn_input_size,
                self._hidden_size,
                num_layers=num_rnn_layers,
                rnn_type=rnn_type,
                trainable_masked_hidden_state=trainable_masked_hidden_state,
            )

        self.state_encoders = nn.ModuleDict(state_encoders)

        self.belief_names = list(self.state_encoders.keys())

        get_logger().info(
            "there are {} belief models: {}".format(
                len(self.belief_names), self.belief_names
            )
        )

    def load_state_dict(self, state_dict, **kwargs):
        new_state_dict = OrderedDict()
        for key in state_dict.keys():
            if "state_encoder." in key:  # old key name
                new_key = key.replace("state_encoder.", "state_encoders.single_belief.")
            elif "goal_visual_encoder.embed_class" in key:
                new_key = key.replace(
                    "goal_visual_encoder.embed_class", "goal_visual_encoder.embed_goal"
                )
            else:
                new_key = key
            new_state_dict[new_key] = state_dict[key]

        return super().load_state_dict(new_state_dict, **kwargs)  # compatible in keys

    def create_actorcritic_head(self):
        self.actor = LinearActorHead(self._hidden_size, self.action_space.n)
        self.critic = LinearCriticHead(self._hidden_size)

    def create_aux_models(self, obs_embed_size: int, action_embed_size: int):
        if self.auxiliary_uuids is None:
            return
        aux_models = OrderedDict()
        for aux_uuid in self.auxiliary_uuids:
            aux_models[aux_uuid] = AuxiliaryModel(
                aux_uuid=aux_uuid,
                action_dim=self.action_space.n,
                obs_embed_dim=obs_embed_size,
                belief_dim=self._hidden_size,
                action_embed_size=action_embed_size,
            )

        self.aux_models = nn.ModuleDict(aux_models)

    @property
    def num_recurrent_layers(self):
        """Number of recurrent hidden layers."""
        return list(self.state_encoders.values())[0].num_recurrent_layers

    @property
    def recurrent_hidden_state_size(self):
        """The recurrent hidden state size of a single model."""
        return self._hidden_size

    def _recurrent_memory_specification(self):
        return {
            memory_key: (
                (
                    ("layer", self.num_recurrent_layers),
                    ("sampler", None),
                    ("hidden", self.recurrent_hidden_state_size),
                ),
                torch.float32,
            )
            for memory_key in self.belief_names
        }

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        raise NotImplementedError("Obs Encoder Not Implemented")

    def fuse_beliefs(
            self, beliefs_dict: Dict[str, torch.FloatTensor], obs_embeds: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        all_beliefs = torch.stack(list(beliefs_dict.values()), dim=-1)  # (T, N, H, k)

        if self.multiple_beliefs:  # call the fusion model
            return self.fusion_model(all_beliefs=all_beliefs, obs_embeds=obs_embeds)
        # single belief
        beliefs = all_beliefs.squeeze(-1)  # (T,N,H)
        return beliefs, None

    def forward(  # type:ignore
            self,
            observations: ObservationType,
            memory: Memory,
            prev_actions: torch.Tensor,
            masks: torch.FloatTensor,
    ) -> Tuple[ActorCriticOutput[DistributionType], Optional[Memory]]:
        """Processes input batched observations to produce new actor and critic
        values. Processes input batched observations (along with prior hidden
        states, previous actions, and masks denoting which recurrent hidden
        states should be masked) and returns an `ActorCriticOutput` object
        containing the model's policy (distribution over actions) and
        evaluation of the current state (value).

        # Parameters
        observations : Batched input observations.
        memory : `Memory` containing the hidden states from initial timepoints.
        prev_actions : Tensor of previous actions taken.
        masks : Masks applied to hidden states. See `RNNStateEncoder`.
        # Returns
        Tuple of the `ActorCriticOutput` and recurrent hidden state.
        """

        # 1.1 use perception model (i.e. encoder) to get observation embeddings
        if "obs_embeds" not in observations:
            obs_embeds = self.forward_encoder(observations)
        else:
            obs_embeds = observations["obs_embeds"]

        if self.use_reward_info:
            # reward weights embedding
            reward_weights_embeds = self.forward_reward_weights_embedder(observations['reward_weights'])
            reward_weights_embeds = reward_weights_embeds.reshape(reward_weights_embeds.shape[0], reward_weights_embeds.shape[1], -1) # 1, 2, 72

        # 1.2 use embedding model to get prev_action embeddings
        if self.prev_action_embedder.input_size == self.action_space.n + 1:
            # In this case we have a unique embedding for the start of an episode
            prev_actions_embeds = self.prev_action_embedder(
                torch.where(
                    condition=0 != masks.view(*prev_actions.shape),
                    input=prev_actions + 1,
                    other=torch.zeros_like(prev_actions),
                )
            )
        else:
            prev_actions_embeds = self.prev_action_embedder(prev_actions)

        if self.use_reward_info:
            joint_embeds = torch.cat((obs_embeds, reward_weights_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)
        else:
            joint_embeds = torch.cat((obs_embeds, prev_actions_embeds), dim=-1)  # (T, N, *)

        # 2. use RNNs to get single/multiple beliefs
        beliefs_dict = {}
        for key, model in self.state_encoders.items():
            beliefs_dict[key], rnn_hidden_states = model(
                joint_embeds, memory.tensor(key), masks
            )
            memory.set_tensor(key, rnn_hidden_states)  # update memory here

        # 3. fuse beliefs for multiple belief models
        beliefs, task_weights = self.fuse_beliefs(
            beliefs_dict, obs_embeds
        )  # fused beliefs

        # 4. prepare output
        extras = (
            {
                aux_uuid: {
                    "beliefs": (
                        beliefs_dict[aux_uuid] if self.multiple_beliefs else beliefs
                    ),
                    "obs_embeds": obs_embeds,
                    "aux_model": (
                        self.aux_models[aux_uuid]
                        if aux_uuid in self.aux_models
                        else None
                    ),
                }
                for aux_uuid in self.auxiliary_uuids
            }
            if self.auxiliary_uuids is not None
            else {}
        )

        if self.multiple_beliefs:
            extras[MultiAuxTaskNegEntropyLoss.UUID] = task_weights

        distributions = self.actor(beliefs)
        values = self.critic(beliefs)
        actor_critic_output = ActorCriticOutput(
            distributions=distributions,
            values=values,
            extras=extras,
        )

        # eval
        # if self.cfg.eval:
        #     torch.save(joint_embeds.cpu(), f'{self.pkl_path}/joint_embeds_{self.counter}.pt')
        #     torch.save(prev_actions.cpu(), f'{self.pkl_path}/prev_actions_{self.counter}.pt')
        #     torch.save(beliefs.cpu(), f'{self.pkl_path}/beliefs_{self.counter}.pt')
        #     torch.save(distributions, f'{self.pkl_path}/distributions_{self.counter}.pt')
        #     torch.save(values.cpu(), f'{self.pkl_path}/values_{self.counter}.pt')
        #     torch.save(observations['goal_object_type_ind'].cpu(), f'{self.pkl_path}/goal_obj_{self.counter}.pt')
        #     torch.save(observations['rel_position_change']['dx_dz_dr'].cpu(),
        #                f'{self.pkl_path}/dx_dz_dr_{self.counter}.pt')
        #     torch.save(observations['rel_position_change']['current_allocentric_position'].cpu(),
        #                f'{self.pkl_path}/current_allocentric_position_{self.counter}.pt')
        #     torch.save(observations['visible_objects'].cpu(), f'{self.pkl_path}/visible_objects_{self.counter}.pt')
        #     torch.save(observations['visible_goal_objects'].cpu(), f'{self.pkl_path}/visible_goal_objects_{self.counter}.pt')
        #     torch.save(observations['total_reward'].cpu(), f'{self.pkl_path}/total_reward_{self.counter}.pt')
        #     torch.save(observations['distance_to_goal'].cpu(),
        #                f'{self.pkl_path}/distance_to_goal_{self.counter}.pt')
        #     torch.save(observations['num_steps_taken'].cpu(), f'{self.pkl_path}/num_steps_taken_{self.counter}.pt')
        #     torch.save(observations['rgb'].cpu(), f'{self.pkl_path}/rgb_{self.counter}.pt')
        #     torch.save(observations['rgb_clip_resnet'].cpu(), f'{self.pkl_path}/rgb_clip_resnet_{self.counter}.pt')
        #     torch.save(observations['valid_moves_forward'].cpu(), f'{self.pkl_path}/valid_moves_forward_{self.counter}.pt')
        #     # torch.save(observations['floor_percentage'].cpu(), f'{self.pkl_path}/floor_percentage_{self.counter}.pt')
        #     torch.save(observations['visited_pos'].cpu(), f'{self.pkl_path}/visited_pos_{self.counter}.pt')
        #     torch.save(observations['visited_room'].cpu(), f'{self.pkl_path}/visited_room_{self.counter}.pt')

        self.counter += 1

        return actor_critic_output, memory
