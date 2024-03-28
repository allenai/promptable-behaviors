"""Baseline models for use in the object navigation task.

Object navigation is currently available as a Task in AI2-THOR and
Facebook's Habitat.
"""
from typing import Optional, List, Dict, cast, Tuple, Sequence

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, reduce, repeat
from omegaconf import DictConfig
import os

from gym.spaces import Dict as SpaceDict

from allenact.utils.model_utils import FeatureEmbedding
from allenact.algorithms.onpolicy_sync.policy import ObservationType
from procthor_visual_nav.codebook_models.visual_nav_models import (
    VisualNavActorCritic,
    VisualNavActorCriticCodebook,
    FusionType,
    RunningMeanStd,
)

class CLIPPCAProcessor:
    def __init__(self,
                keep_dims=2048,
                clip_input_uuid=""):
        self.clip_input_uuid = clip_input_uuid
        self.keep_dims = keep_dims
        self.components_T = torch.load("/net/nfs.cirrascale/prior/yuxuanl/planrep/storage/clip_pca/explore_house_v0.8/7x7_components_T.pt")
        if keep_dims == 0:
            # set component weights to zero and pretend keep_dims is 1
            self.components_T = torch.zeros_like(self.components_T[..., :1])
            self.keep_dims = 1
        else:
            self.components_T = self.components_T[..., :keep_dims]  # truncate to preserve only some dimensions

    def process(self, obs):
        clip_features = obs[self.clip_input_uuid]  # (n_frames, 2048, 7, 7)
        assert clip_features.shape[-3:] == (2048, 7, 7)  # TODO: what's the best assert statement?

        # center features
        clip_features = clip_features - clip_features.mean(0)
        # transform features
        self.components_T = self.components_T.to(clip_features.device)
        transformed = torch.matmul(clip_features.permute(2, 3, 0, 1),  # (7, 7, n_frames, 2048)
                                   self.components_T)
        transformed = transformed.permute(2, 3, 0, 1)  # (n_frames, keep_dims, 7, 7)

        return transformed
class CatObservations(nn.Module):
    def __init__(self, ordered_uuids: Sequence[str], dim: int):
        super().__init__()
        assert len(ordered_uuids) != 0

        self.ordered_uuids = ordered_uuids
        self.dim = dim

    def forward(self, observations: ObservationType):
        if len(self.ordered_uuids) == 1:
            return observations[self.ordered_uuids[0]]
        return torch.cat(
            [observations[uuid] for uuid in self.ordered_uuids], dim=self.dim
        )


class ResnetTensorNavActorCritic(VisualNavActorCritic):
    def __init__(
            # base params
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=False,
            add_prev_action_null_token=False,
            action_embed_size=6,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[List[str]] = None,
            # custom params
            rgb_resnet_preprocessor_uuid: Optional[str] = None,
            depth_resnet_preprocessor_uuid: Optional[str] = None,
            goal_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
            # morl params
            num_objectives=4,
            save_obs_embeds=False,
            use_reward_info=False,
            **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
            **kwargs,
        )

        if (
                rgb_resnet_preprocessor_uuid is None
                or depth_resnet_preprocessor_uuid is None
        ):
            resnet_preprocessor_uuid = (
                rgb_resnet_preprocessor_uuid
                if rgb_resnet_preprocessor_uuid is not None
                else depth_resnet_preprocessor_uuid
            )

            if goal_sensor_uuid is None:
                # no goal, only visual encoder
                self.goal_visual_encoder = ResnetTensorEncoder(
                    self.observation_space,
                    resnet_preprocessor_uuid,
                    resnet_compressor_hidden_out_dims,
                    combiner_hidden_out_dims,
                )
            else:
                self.goal_visual_encoder = ResnetTensorGoalEncoder(
                    self.observation_space,
                    goal_sensor_uuid,
                    resnet_preprocessor_uuid,
                    goal_dims,
                    resnet_compressor_hidden_out_dims,
                    combiner_hidden_out_dims,
                    **kwargs,
                )
        else:
            self.goal_visual_encoder = ResnetDualTensorGoalEncoder(  # type:ignore
                self.observation_space,
                goal_sensor_uuid,
                rgb_resnet_preprocessor_uuid,
                depth_resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )

        self.use_reward_info = use_reward_info
        if use_reward_info:
            self.create_reward_weights_embedder(num_objectives=num_objectives, embed_size=action_embed_size*2*num_objectives)
            reward_info_embed_size = 2*action_embed_size # per objective
        else:
            reward_info_embed_size = 0

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
            reward_info_embed_size=reward_info_embed_size,
            num_objectives=num_objectives,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.save_obs_embeds = save_obs_embeds

        self.train()

    def create_reward_weights_embedder(self, num_objectives: int, embed_size: int, mode='codebook'):
        if mode == 'codebook':
            self.reward_weights_embedder = CodebookEmbedder(input_dim=num_objectives, embedding_dim=embed_size) # prev_action_embed_size*2
        elif mode == 'embedding':
            self.reward_weights_embedder = FeatureEmbedding(input_dim=30, output_size=embed_size) # prev_action_embed_size*2
        elif mode == 'raw':
            self.reward_weights_embedder = nn.Linear(num_objectives, embed_size)

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)
    
    def forward_reward_weights_embedder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.reward_weights_embedder(observations)


class CodebookEmbedder(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=20, codebook_dim=30, embedding_dim=48):
        super(CodebookEmbedder, self).__init__()
        
        # The encoder consists of two linear layers separated by a ReLU activation.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, codebook_dim)
        )
        
        # The codebook contains 30 different vectors, each of 48 dimensions.
        self.codebook = nn.Parameter(torch.randn(codebook_dim, embedding_dim))

    def forward(self, x):
        # Transform the input to a distribution over the codebook vectors
        # x shape: W x B x num_objectives
        x_shape = x.shape
        x = x.view(-1, x_shape[-1]).type(torch.float32)
        logits = self.encoder(x)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Use the distribution to get the codebook embeddings
        embeddings = torch.matmul(probs, self.codebook)
        embeddings = embeddings.view(x_shape[0], x_shape[1], -1)
        return embeddings


class ResnetTensorNavActorCriticCodebook(VisualNavActorCriticCodebook):
    def __init__(
            # base params
            self,
            action_space: gym.spaces.Discrete,
            observation_space: SpaceDict,
            goal_sensor_uuid: str,
            hidden_size=512,
            num_rnn_layers=1,
            rnn_type="GRU",
            add_prev_actions=False,
            add_prev_action_null_token=False,
            action_embed_size=6,
            multiple_beliefs=False,
            beliefs_fusion: Optional[FusionType] = None,
            auxiliary_uuids: Optional[List[str]] = None,
            # custom params
            rgb_resnet_preprocessor_uuid: Optional[str] = None,
            depth_resnet_preprocessor_uuid: Optional[str] = None,
            goal_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
            **kwargs,
    ):
        super().__init__(
            action_space=action_space,
            observation_space=observation_space,
            hidden_size=hidden_size,
            multiple_beliefs=multiple_beliefs,
            beliefs_fusion=beliefs_fusion,
            auxiliary_uuids=auxiliary_uuids,
            **kwargs,
        )

        if (
                rgb_resnet_preprocessor_uuid is None
                or depth_resnet_preprocessor_uuid is None
        ):
            resnet_preprocessor_uuid = (
                rgb_resnet_preprocessor_uuid
                if rgb_resnet_preprocessor_uuid is not None
                else depth_resnet_preprocessor_uuid
            )
            self.goal_visual_encoder = ResnetTensorGoalEncoder(
                self.observation_space,
                goal_sensor_uuid,
                resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
                **kwargs,
            )
        else:
            self.goal_visual_encoder = ResnetDualTensorGoalEncoder(  # type:ignore
                self.observation_space,
                goal_sensor_uuid,
                rgb_resnet_preprocessor_uuid,
                depth_resnet_preprocessor_uuid,
                goal_dims,
                resnet_compressor_hidden_out_dims,
                combiner_hidden_out_dims,
            )

        self.create_state_encoders(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            num_rnn_layers=num_rnn_layers,
            rnn_type=rnn_type,
            add_prev_actions=add_prev_actions,
            add_prev_action_null_token=add_prev_action_null_token,
            prev_action_embed_size=action_embed_size,
        )

        self.create_actorcritic_head()

        self.create_aux_models(
            obs_embed_size=self.goal_visual_encoder.output_dims,
            action_embed_size=action_embed_size,
        )

        self.train()

    @property
    def is_blind(self) -> bool:
        """True if the model is blind (e.g. neither 'depth' or 'rgb' is an
        input observation type)."""
        return self.goal_visual_encoder.is_blind

    def forward_encoder(self, observations: ObservationType) -> torch.FloatTensor:
        return self.goal_visual_encoder(observations)


class ResnetTensorGoalEncoder(nn.Module):
    def __init__(
            self,
            observation_spaces: SpaceDict,
            goal_sensor_uuid: str,
            resnet_preprocessor_uuid: str,
            goal_embed_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
            cfg: DictConfig = None,
    ) -> None:
        super().__init__()

        self.counter = 0
        self.cfg = cfg
        self.logging_path = f'{cfg.output_dir}/logs/{cfg.wandb.name}'
        self.pkl_path = f'{cfg.output_dir}/pkls/{cfg.wandb.name}'
        if not os.path.exists(self.logging_path):
            os.makedirs(self.logging_path)
        if not os.path.exists(self.pkl_path):
            os.makedirs(self.pkl_path)
        self.goal_uuid = goal_sensor_uuid
        self.resnet_uuid = resnet_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]
        if isinstance(self.goal_space, gym.spaces.Discrete):
            self.embed_goal = nn.Embedding(
                num_embeddings=self.goal_space.n, embedding_dim=self.goal_embed_dims,
            )
        elif isinstance(self.goal_space, gym.spaces.Box):
            self.embed_goal = nn.Linear(self.goal_space.shape[-1], self.goal_embed_dims)
        else:
            raise NotImplementedError

        self.blind = self.resnet_uuid not in observation_spaces.spaces
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuid].shape
            self.resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

        # codebook
        if cfg.model.codebook.embeds == "obs_embeds":
            self.codebook_type = cfg.model.codebook.type
            self.codebook_size = cfg.model.codebook.size
            self.code_dim = cfg.model.codebook.code_dim

            if self.cfg.model.codebook.initialization == "dictionary_learning":
                codebook_init = torch.tensor(np.load(f'{cfg.output_dir}/basis_vectors_baseline_5k_frames.npy'),
                                             dtype=torch.float32).contiguous()
                codebook_init_normalized = -1 + 2 * (codebook_init - codebook_init.min()) / (
                            codebook_init.max() - codebook_init.min())
                self.codebook = torch.nn.Parameter(torch.Tensor(self.codebook_size, self.code_dim))
                self.codebook.data = codebook_init_normalized
            elif self.cfg.model.codebook.initialization == "random":
                self.codebook = torch.nn.Parameter(torch.randn(self.codebook_size, self.code_dim))
            self.codebook.requires_grad = True
            # dropout to prevent codebook collapse
            self.dropout_prob = cfg.model.codebook.dropout
            self.dropout = nn.Dropout(self.dropout_prob)

            # codebook indexing
            self.linear_codebook_indexer = nn.Sequential(
                nn.ReLU(),
                nn.Linear(1568, self.codebook_size),
            )
            if self.codebook_type == "softmax_only":
                self.linear_upsample = nn.Sequential(
                    nn.Linear(1568, 1568),
                )
            else:
                self.linear_upsample = nn.Sequential(
                    nn.Linear(self.code_dim, 1568),
                )

            if self.codebook_type == "linear":
                # linear layer instead of codebook
                self.linear_layer_codebook = nn.Sequential(
                    nn.Linear(self.codebook_size, self.code_dim),
                )

            # running mean and std
            self.rms = RunningMeanStd()

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return (
                    self.combine_hid_out_dims[-1]
                    * self.resnet_tensor_shape[1]
                    * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        observations = {**observations}
        resnet = observations[self.resnet_uuid]
        goal = observations[self.goal_uuid]

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        else:
            nstep, nsampler = resnet.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])
        observations[self.goal_uuid] = goal.view(-1, goal.shape[-1])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])

        obs_embs = self.compress_resnet(observations)
        goal_embs = self.distribute_target(observations)

        if self.cfg.model.codebook.embeds == "obs_embeds":
            obs_embs_flat = obs_embs.view(obs_embs.shape[0], -1)
            if self.codebook_type == "softmax_only":
                codebook_probs = F.softmax(obs_embs_flat, dim=-1)
            else:
                codebook_probs = F.softmax(self.linear_codebook_indexer(obs_embs_flat), dim=-1)

            if self.dropout_prob > 0:
                codebook_probs = self.dropout(codebook_probs)

            self.rms.update(codebook_probs.detach().cpu().numpy())
            if self.codebook_type == "learned":
                code_output = torch.einsum('nm,md->nd', codebook_probs, self.codebook)
            elif self.codebook_type == "random":
                code_output = torch.einsum('nm,md->nd', codebook_probs, self.codebook.detach())
            elif self.codebook_type == "binary":
                code_output = torch.einsum('nm,md->nd', codebook_probs, 2.0 * (self.codebook > 0.) - 1)
            elif self.codebook_type == "linear":
                code_output = self.linear_layer_codebook(codebook_probs)
            elif self.codebook_type == "softmax_only":
                code_output = codebook_probs

            obs_embs_flat = self.linear_upsample(code_output)
            obs_embs = obs_embs_flat.view(obs_embs.shape)

            if not self.cfg.eval and self.counter > 0 and self.counter % self.cfg.logging.frequency == 0:
                torch.save(self.codebook.detach().cpu(), f'{self.logging_path}/codebook_{self.counter}.pt')
                torch.save(self.codebook.detach().cpu(), f'{self.logging_path}/codebook_latest.pt')
                np.save(f'{self.logging_path}/codebook_probs_{self.counter}.npy', self.rms.mean)
                np.save(f'{self.logging_path}/codebook_probs_latest.npy', self.rms.mean)
            self.counter += 1

        embs = [
            obs_embs,
            goal_embs,
        ]
        x = self.target_obs_combiner(torch.cat(embs, dim=1, ))
        x = x.reshape(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)


class ResnetDualTensorGoalEncoder(nn.Module):
    def __init__(
            self,
            observation_spaces: SpaceDict,
            goal_sensor_uuid: str,
            rgb_resnet_preprocessor_uuid: str,
            depth_resnet_preprocessor_uuid: str,
            goal_embed_dims: int = 32,
            resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
            combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.goal_uuid = goal_sensor_uuid
        self.rgb_resnet_uuid = rgb_resnet_preprocessor_uuid
        self.depth_resnet_uuid = depth_resnet_preprocessor_uuid
        self.goal_embed_dims = goal_embed_dims
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.goal_space = observation_spaces.spaces[self.goal_uuid]
        if isinstance(self.goal_space, gym.spaces.Discrete):
            self.embed_goal = nn.Embedding(
                num_embeddings=self.goal_space.n, embedding_dim=self.goal_embed_dims,
            )
        elif isinstance(self.goal_space, gym.spaces.Box):
            self.embed_goal = nn.Linear(self.goal_space.shape[-1], self.goal_embed_dims)
        else:
            raise NotImplementedError

        self.blind = (
                self.rgb_resnet_uuid not in observation_spaces.spaces
                or self.depth_resnet_uuid not in observation_spaces.spaces
        )
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[
                self.rgb_resnet_uuid
            ].shape
            self.rgb_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.depth_resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )
            self.rgb_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )
            self.depth_target_obs_combiner = nn.Sequential(
                nn.Conv2d(
                    self.resnet_hid_out_dims[1] + self.goal_embed_dims,
                    self.combine_hid_out_dims[0],
                    1,
                ),
                nn.ReLU(),
                nn.Conv2d(*self.combine_hid_out_dims[0:2], 1),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return (
                    2
                    * self.combine_hid_out_dims[-1]
                    * self.resnet_tensor_shape[1]
                    * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
            self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_rgb_resnet(self, observations):
        return self.rgb_resnet_compressor(observations[self.rgb_resnet_uuid])

    def compress_depth_resnet(self, observations):
        return self.depth_resnet_compressor(observations[self.depth_resnet_uuid])

    def distribute_target(self, observations):
        target_emb = self.embed_goal(observations[self.goal_uuid])
        return target_emb.view(-1, self.goal_embed_dims, 1, 1).expand(
            -1, -1, self.resnet_tensor_shape[-2], self.resnet_tensor_shape[-1]
        )

    def adapt_input(self, observations):
        rgb = observations[self.rgb_resnet_uuid]
        depth = observations[self.depth_resnet_uuid]

        use_agent = False
        nagent = 1

        if len(rgb.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = rgb.shape[:3]
        else:
            nstep, nsampler = rgb.shape[:2]

        observations[self.rgb_resnet_uuid] = rgb.view(-1, *rgb.shape[-3:])
        observations[self.depth_resnet_uuid] = depth.view(-1, *depth.shape[-3:])
        observations[self.goal_uuid] = observations[self.goal_uuid].view(-1, 1)

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        rgb_embs = [
            self.compress_rgb_resnet(observations),
            self.distribute_target(observations),
        ]
        rgb_x = self.rgb_target_obs_combiner(torch.cat(rgb_embs, dim=1, ))
        depth_embs = [
            self.compress_depth_resnet(observations),
            self.distribute_target(observations),
        ]
        depth_x = self.depth_target_obs_combiner(torch.cat(depth_embs, dim=1, ))
        x = torch.cat([rgb_x, depth_x], dim=1)
        x = x.reshape(x.shape[0], -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)

class ResnetTensorEncoder(nn.Module):
    def __init__(
        self,
        observation_spaces: SpaceDict,
        resnet_preprocessor_uuid: str,
        resnet_compressor_hidden_out_dims: Tuple[int, int] = (128, 32),
        combiner_hidden_out_dims: Tuple[int, int] = (128, 32),
    ) -> None:
        super().__init__()
        self.resnet_uuid = resnet_preprocessor_uuid
        self.resnet_hid_out_dims = resnet_compressor_hidden_out_dims
        self.combine_hid_out_dims = combiner_hidden_out_dims

        self.blind = self.resnet_uuid not in observation_spaces.spaces
        if not self.blind:
            self.resnet_tensor_shape = observation_spaces.spaces[self.resnet_uuid].shape
            self.resnet_compressor = nn.Sequential(
                nn.Conv2d(self.resnet_tensor_shape[0], self.resnet_hid_out_dims[0], 1),
                nn.ReLU(),
                nn.Conv2d(*self.resnet_hid_out_dims[0:2], 1),
                nn.ReLU(),
            )

    @property
    def is_blind(self):
        return self.blind

    @property
    def output_dims(self):
        if self.blind:
            return self.goal_embed_dims
        else:
            return (
                self.resnet_hid_out_dims[-1]
                * self.resnet_tensor_shape[1]
                * self.resnet_tensor_shape[2]
            )

    def get_object_type_encoding(
        self, observations: Dict[str, torch.FloatTensor]
    ) -> torch.FloatTensor:
        """Get the object type encoding from input batched observations."""
        return cast(
            torch.FloatTensor,
            self.embed_goal(observations[self.goal_uuid].to(torch.int64)),
        )

    def compress_resnet(self, observations):
        return self.resnet_compressor(observations[self.resnet_uuid])

    def adapt_input(self, observations):
        resnet = observations[self.resnet_uuid]

        use_agent = False
        nagent = 1

        if len(resnet.shape) == 6:
            use_agent = True
            nstep, nsampler, nagent = resnet.shape[:3]
        elif len(resnet.shape) == 5:
            nstep, nsampler = resnet.shape[:2]

        observations[self.resnet_uuid] = resnet.view(-1, *resnet.shape[-3:])

        return observations, use_agent, nstep, nsampler, nagent

    @staticmethod
    def adapt_output(x, use_agent, nstep, nsampler, nagent):
        if use_agent:
            return x.view(nstep, nsampler, nagent, -1)
        return x.view(nstep, nsampler * nagent, -1)

    def forward(self, observations):
        observations, use_agent, nstep, nsampler, nagent = self.adapt_input(
            observations
        )

        if self.blind:
            return self.embed_goal(observations[self.goal_uuid])
        x = self.compress_resnet(observations)
        x = x.reshape(x.size(0), -1)  # flatten

        return self.adapt_output(x, use_agent, nstep, nsampler, nagent)