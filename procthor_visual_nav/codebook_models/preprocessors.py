import gym
import torch

from allenact.base_abstractions.preprocessor import Preprocessor


class ClipPCAPreprocessor(Preprocessor):
    '''
    transforms clip features via pre-computed PCA
    requires that components be pre-fitted against a sample of clip features (e.g. of training data)
    and the transposed components saved
    '''

    def __init__(self,
                 clip_input_uuid: str,
                 output_uuid: str,
                 components_ckpt: str,
                 clip_observation_space: gym.spaces,
                 keep_dims: int = 2048,
                 **kwargs):

        self.clip_input_uuid = clip_input_uuid
        input_uuids = [clip_input_uuid]
        assert (len(input_uuids) == 1), "clip PCA preprocessor can only consume one observation type"

        self.keep_dims = keep_dims
        self.components_T = torch.load(components_ckpt)
        if keep_dims == 0:
            # set component weights to zero and pretend keep_dims is 1
            self.components_T = torch.zeros_like(self.components_T[..., :1])
            self.keep_dims = 1
        else:
            self.components_T = self.components_T[..., :keep_dims]  # truncate to preserve only some dimensions

        # truncate from clip_observation_space
        observation_space = gym.spaces.Box(
            low=clip_observation_space.low[:self.keep_dims],
            high=clip_observation_space.high[:self.keep_dims],
            shape=(self.keep_dims, *clip_observation_space.shape[1:]),
        )

        super().__init__(input_uuids=input_uuids,
                         output_uuid=output_uuid,
                         observation_space=observation_space)

    def to(self, device):
        self.components_T = self.components_T.to(device)
        self.device = device
        return self

    def process(self, obs):

        clip_features = obs[self.clip_input_uuid]  # (n_frames, 2048, 7, 7)

        assert clip_features.shape[-3:] == (2048, 7, 7)  # TODO: what's the best assert statement?

        # center features
        clip_features = clip_features - clip_features.mean(0)
        # transform features
        transformed = torch.matmul(clip_features.permute(2, 3, 0, 1),  # (7, 7, n_frames, 2048)
                                   self.components_T)
        transformed = transformed.permute(2, 3, 0, 1)  # (n_frames, keep_dims, 7, 7)

        return transformed