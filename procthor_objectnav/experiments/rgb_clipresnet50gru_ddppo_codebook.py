try:
    from typing import final
except ImportError:
    from typing_extensions import final

import gym
import torch
import torch.nn as nn
from allenact_plugins.ithor_plugin.ithor_sensors import GoalObjectTypeThorSensor
from allenact.embodiedai.sensors.vision_sensors import DepthSensor, RGBSensor

from procthor_objectnav.codebook_models.objectnav_models import ResnetTensorNavActorCriticCodebook
from procthor_objectnav.experiments.rgb_clipresnet50gru_ddppo import ProcTHORObjectNavRGBClipResNet50PPOExperimentConfig
from ..sensors.sensors import *
from ..sensors.vision import RGBSensorThorController



class ProcTHORObjectNavRGBClipResNet50PPOCodeBookExperimentConfig(ProcTHORObjectNavRGBClipResNet50PPOExperimentConfig):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

        self.MODEL = ResnetTensorNavActorCriticCodebook

    @classmethod
    def tag(cls):
        return "ObjectNav-RGB-ClipResNet50GRU-DDPPO-CodeBook"



