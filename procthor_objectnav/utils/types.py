from typing import Any, Dict, List, Optional, Tuple

try:
    from typing import Literal, TypedDict
except ImportError:
    from typing_extensions import Literal, TypedDict

from allenact.base_abstractions.sensor import Sensor
from attrs import define


class Vector3(TypedDict):
    x: float
    y: float
    z: float


@define
class TaskSamplerArgs:
    process_ind: int
    """The process index number."""

    mode: Literal["train", "eval"]
    """Whether we are in training or evaluation mode."""

    house_inds: List[int]
    """Which houses to use for each process."""

    houses: Any
    """The hugging face Dataset of all the houses in the split."""

    sensors: List[Sensor]
    """The sensors to use for each task."""

    controller_args: Dict[str, Any]
    """The arguments to pass to the AI2-THOR controller."""

    reward_config: Dict[str, Any]
    """The reward configuration to use."""

    target_object_types: List[str]
    """The object types to use as targets."""

    max_steps: int
    """The maximum number of steps to run each task."""

    max_tasks: int
    """The maximum number of tasks to run."""

    distance_type: str
    """The type of distance computation to use ("l2" or "geo")."""

    resample_same_scene_freq: int
    """
    Number of times to sample a scene/house before moving to the next one.
    
    If <1 then will never 
        sample a new scene (unless `force_advance_scene=True` is passed to `next_task(...)`.
    ."""

    p_randomize_materials: float
    test_on_validation: bool
    actions: Tuple[str]
    max_agent_positions: int
    max_vis_points: int
    p_greedy_target_object: float
    ithor_p_shuffle_objects: float
    valid_agent_heights: float

    # Can we remove?
    deterministic_cudnn: bool = False
    loop_dataset: bool = True
    seed: Optional[int] = None
    allow_flipping: bool = False

    # morl
    objectives_combination: Optional[str] = None
    objectives: Optional[List[str]] = None
    valid_objectives: Optional[List[str]] = None
    adaptive_reward: Optional[bool] = False
    reward_embed_type: Optional[str] = "codebook"
    reward_weights: Optional[List[float]] = None
    save_trajectory: Optional[bool] = False
    save_video: Optional[bool] = False


@define
class RewardConfig:
    step_penalty: float
    shaping_weight: float
    reached_horizon_reward: float
    positive_only_reward: bool
    exploration_reward: float
    safety_reward_threshold: float
    safety_reward_scale: float
    safety_distance: float
    far_from_initial_scale: float
    reward_grid_size: float
    # optional
    goal_success_reward: Optional[float] = None
    failed_stop_reward: Optional[float] = None
    object_found: Optional[float] = None
    collision_penalty: Optional[float] = None



class AgentPose(TypedDict):
    position: Vector3
    rotation: Vector3
    horizon: int
    standing: bool
