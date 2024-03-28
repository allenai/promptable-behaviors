from typing import Any, Dict, Optional, Union, Sequence

import numpy as np
import gym

from allenact.base_abstractions.sensor import Sensor
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from allenact_plugins.robothor_plugin.robothor_environment import RoboThorEnvironment
from allenact_plugins.robothor_plugin.robothor_tasks import PointNavTask, ObjectNavTask
from allenact_plugins.ithor_plugin.ithor_tasks import ObjectNaviThorGridTask
from allenact.utils.misc_utils import prepare_locals_for_super
from allenact.base_abstractions.task import Task

from ai2thor.controller import Controller


ALL_OBJECTS = [
    # A
    "AlarmClock", "AluminumFoil", "Apple", "AppleSliced", "ArmChair",
    "BaseballBat", "BasketBall", "Bathtub", "BathtubBasin", "Bed", "Blinds", "Book", "Boots", "Bottle", "Bowl", "Box",
    # B
    "Bread", "BreadSliced", "ButterKnife",
    # C
    "Cabinet", "Candle", "CD", "CellPhone", "Chair", "Cloth", "CoffeeMachine", "CoffeeTable", "CounterTop",
    "CreditCard",
    "Cup", "Curtains",
    # D
    "Desk", "DeskLamp", "Desktop", "DiningTable", "DishSponge", "DogBed", "Drawer", "Dresser", "Dumbbell",
    # E
    "Egg", "EggCracked",
    # F
    "Faucet", "Floor", "FloorLamp", "Footstool", "Fork", "Fridge",
    # G
    "GarbageBag", "GarbageCan",
    # H
    "HandTowel", "HandTowelHolder", "HousePlant", "Kettle", "KeyChain", "Knife",
    # L
    "Ladle", "Laptop", "LaundryHamper", "Lettuce", "LettuceSliced", "LightSwitch",
    # M
    "Microwave", "Mirror", "Mug",
    # N
    "Newspaper",
    # O
    "Ottoman",
    # P
    "Painting", "Pan", "PaperTowel", "Pen", "Pencil", "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot",
    "Potato", "PotatoSliced",
    # R
    "RemoteControl", "RoomDecor",
    # S
    "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShelvingUnit", "ShowerCurtain", "ShowerDoor", "ShowerGlass",
    "ShowerHead", "SideTable", "Sink", "SinkBasin", "SoapBar", "SoapBottle", "Sofa", "Spatula", "Spoon", "SprayBottle",
    "Statue", "Stool", "StoveBurner", "StoveKnob",
    # T
    "TableTopDecor", "TargetCircle", "TeddyBear", "Television", "TennisRacket", "TissueBox", "Toaster", "Toilet",
    "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced", "Towel", "TowelHolder", "TVStand",
    # V
    "VacuumCleaner", "Vase",
    # W
    "Watch", "WateringCan", "Window", "WineBottle",
]

SELECTED_OBJECTS = [
    # A
    "AlarmClock", "Apple", "ArmChair",
    "BaseballBat", "BasketBall", "Bathtub", "Bed", "Blinds", "Book", "Boots", "Bottle", "Bowl", "Box",
    # B
    "Bread", "ButterKnife",
    # C
    "Cabinet", "Candle", "CD", "CellPhone", "Chair", "Cloth", "CoffeeMachine", "CoffeeTable", "CounterTop",
    "CreditCard",
    "Cup", "Curtains",
    # D
    "Desk", "DeskLamp", "Desktop", "DiningTable", "DishSponge", "DogBed", "Drawer", "Dresser", "Dumbbell",
    # E
    "Egg",
    # F
    "Faucet", "Floor", "FloorLamp", "Footstool", "Fork", "Fridge",
    # G
    "GarbageBag", "GarbageCan",
    # H
    "HandTowel", "HousePlant", "Kettle", "KeyChain", "Knife",
    # L
    "Ladle", "Laptop", "LaundryHamper", "Lettuce",
    # M
    "Microwave", "Mirror", "Mug",
    # N
    "Newspaper",
    # O
    "Ottoman",
    # P
    "Painting", "Pan", "PaperTowel", "Pen", "Pencil", "PepperShaker", "Pillow", "Plate", "Plunger", "Poster", "Pot",
    "Potato",
    # R
    "RemoteControl", "RoomDecor",
    # S
    "Safe", "SaltShaker", "ScrubBrush", "Shelf", "ShowerCurtain", "ShowerDoor", "ShowerGlass",
    "ShowerHead", "SideTable", "Sink", "SoapBar", "SoapBottle", "Sofa", "Spatula", "Spoon", "SprayBottle",
    "Statue", "Stool", "StoveBurner", "StoveKnob",
    # T
    "TableTopDecor", "TargetCircle", "TeddyBear", "Television", "TennisRacket", "TissueBox", "Toaster", "Toilet",
    "ToiletPaper", "ToiletPaperHanger", "Tomato", "TomatoSliced", "Towel", "TowelHolder", "TVStand",
    # V
    "VacuumCleaner", "Vase",
    # W
    "Watch", "WateringCan", "Window", "WineBottle",
]

TARGET_TYPES = tuple(
    sorted(
        [
            'AlarmClock',
            'Apple',
            'BaseballBat',
            'BasketBall',
            'Bed',
            'Bowl',
            'Chair',
            'GarbageCan',
            'HousePlant',
            'Laptop',
            'Mug',
            'Sofa',
            'SprayBottle',
            'Television',
            'Toilet',
            'Vase'
        ]
    )
)


class DistanceToGoalSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = gym.spaces.Dict(
            {
                "distance_to_goal": gym.spaces.Box(
                    low=np.array([-np.inf], dtype=np.float32),
                    high=np.array([np.inf], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(  # type:ignore
            self,
            env: IThorEnvironment,
            task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
            *args,
            **kwargs,
    ) -> np.ndarray:
        return task.dist_to_target_func()
        # return np.array([task.dist_to_target_func()], dtype=np.float32)


class TotalRewardSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = gym.spaces.Dict(
            {
                "total_reward": gym.spaces.Box(
                    low=np.array([-np.inf], dtype=np.float32),
                    high=np.array([np.inf], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(  # type:ignore
            self,
            env: IThorEnvironment,
            task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
            *args,
            **kwargs,
    ) -> np.ndarray:
        # return np.array([np.sum(task._rewards)], dtype=np.float32)
        return np.sum(task._rewards)


class NumStepsTakenSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, max_steps: int, uuid: str, **kwargs: Any) -> None:
        observation_space = gym.spaces.Discrete(max_steps)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(  # type:ignore
            self,
            env: IThorEnvironment,
            task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
            *args,
            **kwargs,
    ) -> np.ndarray:
        # return np.array([task.num_steps_taken()], dtype=np.int64)
        return task.num_steps_taken()


class ActionSuccessSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self) -> gym.spaces.Discrete:
        """The observation space.

        Equals `gym.spaces.Discrete(2)` where a 0 indicates that the agent
        **should not** take the `End` action and a 1 indicates that the agent
        **should** take the end action.
        """
        return gym.spaces.Discrete(2)

    def get_observation(  # type:ignore
            self,
            env: IThorEnvironment,
            task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
            *args,
            **kwargs,
    ) -> np.ndarray:
        action_success = env.last_event.metadata["lastActionSuccess"]
        return np.array([1. * action_success], dtype=np.int64)


class RelativePositionChangeTHORSensor(
    Sensor[RoboThorEnvironment, Task[RoboThorEnvironment]]
):
    def __init__(self, uuid: str = "rel_position_change", **kwargs: Any):
        observation_space = gym.spaces.Dict(
            {
                "last_allocentric_position": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, 360], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "dx_dz_dr": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, -360], dtype=np.float32),
                    high=np.array([-np.inf, -np.inf, 360], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32,
                ),
                "current_allocentric_position": gym.spaces.Box(
                    low=np.array([-np.inf, -np.inf, 0], dtype=np.float32),
                    high=np.array([np.inf, np.inf, 360], dtype=np.float32),
                    shape=(3,),
                    dtype=np.float32,
                ),
            }
        )
        super().__init__(**prepare_locals_for_super(locals()))

        self.last_xzr: Optional[np.ndarray] = None

    @staticmethod
    def get_relative_position_change(from_xzr: np.ndarray, to_xzr: np.ndarray):
        dx_dz_dr = to_xzr - from_xzr

        # Transform dx, dz (in global coordinates) into the relative coordinates
        # given by rotation r0=from_xzr[-2]. This requires rotating everything so that
        # r0 is facing in the positive z direction. Since thor rotations are negative
        # the usual rotation direction this means we want to rotate by r0 degrees.
        theta = np.pi * from_xzr[-1] / 180
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        dx_dz_dr = (
                np.array(
                    [
                        [cos_theta, -sin_theta, 0],
                        [sin_theta, cos_theta, 0],
                        [0, 0, 1],  # Don't change dr
                    ]
                )
                @ dx_dz_dr.reshape(-1, 1)
        ).reshape(-1)

        dx_dz_dr[-1] = (dx_dz_dr[-1] % 360)
        return dx_dz_dr

    def get_observation(
            self,
            env: RoboThorEnvironment,
            task: Optional[Task[RoboThorEnvironment]],
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        if task.num_steps_taken() == 0:
            p = env.last_event.metadata["agent"]["position"]
            r = env.last_event.metadata["agent"]["rotation"]["y"]
            self.last_xzr = np.array([p["x"], p["z"], r % 360])

        p = env.last_event.metadata["agent"]["position"]
        r = env.last_event.metadata["agent"]["rotation"]["y"]
        current_xzr = np.array([p["x"], p["z"], r % 360])

        dx_dz_dr = self.get_relative_position_change(
            from_xzr=self.last_xzr, to_xzr=current_xzr
        )

        to_return = {"last_allocentric_position": self.last_xzr, "dx_dz_dr": dx_dz_dr,
                     "current_allocentric_position": current_xzr}

        self.last_xzr = current_xzr

        return to_return


class GoalObjectTypeThorSensor(Sensor):
    def __init__(
            self,
            object_types: Sequence[str],
            target_to_detector_map: Optional[Dict[str, str]] = None,
            detector_types: Optional[Sequence[str]] = None,
            uuid: str = "goal_object_type_ind",
            **kwargs: Any,
    ):
        self.ordered_object_types = list(object_types)
        assert self.ordered_object_types == sorted(
            self.ordered_object_types
        ), "object types input to goal object type sensor must be ordered"

        self.target_to_detector_map = target_to_detector_map

        if target_to_detector_map is None:
            self.object_type_to_ind = {
                ot: i for i, ot in enumerate(self.ordered_object_types)
            }
        else:
            assert (
                    detector_types is not None
            ), "Missing detector_types for map {}".format(target_to_detector_map)
            self.target_to_detector = target_to_detector_map
            self.detector_types = detector_types

            detector_index = {ot: i for i, ot in enumerate(self.detector_types)}
            self.object_type_to_ind = {
                ot: detector_index[self.target_to_detector[ot]]
                for ot in self.ordered_object_types
            }

        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        if self.target_to_detector_map is None:
            return gym.spaces.Discrete(len(self.ordered_object_types))
        else:
            return gym.spaces.Discrete(len(self.detector_types))

    def get_observation(
            self,
            env: IThorEnvironment,
            task: Optional[ObjectNaviThorGridTask],
            *args: Any,
            **kwargs: Any,
    ) -> Any:
        return self.object_type_to_ind[task.task_info["object_type"]]


class VisibleObjectTypesSensor(
    Sensor[
        Union[IThorEnvironment, RoboThorEnvironment],
        Union[Task[IThorEnvironment], Task[RoboThorEnvironment]],
    ]
):
    def __init__(self, uuid: str = "visible_objects", **kwargs: Any):
        super().__init__(
            uuid=uuid,
            observation_space=gym.spaces.Box(
                low=0, high=1, shape=(len(ALL_OBJECTS),)
            ),
            **kwargs
        )
        self.type_to_index = {
            tt: i for i, tt in enumerate(ALL_OBJECTS)
        }

    def get_observation(
            self,
            env: Union[IThorEnvironment, RoboThorEnvironment],
            task: Optional[Task],
            *args: Any,
            **kwargs: Any
    ) -> Any:
        out = np.zeros((len(self.type_to_index),))
        for o in env.last_event.metadata["objects"]:
            if o["visible"] and o["objectType"] in self.type_to_index:
                out[self.type_to_index[o["objectType"]]] = 1.0
        return out


class VisibilitySensor(Sensor):
    def __init__(self, uuid, **kwargs):
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(ALL_OBJECTS),),
            dtype=np.int64,
        )

        self.visible_distance = 5.0
        self.entity_types = ALL_OBJECTS

        super().__init__(**prepare_locals_for_super(locals()))

        self.uuid = uuid

    def get_observation(self, env, task, *args: Any, **kwargs: Any) -> Any:
        last_seen_object_ids = set(
            env.step(
                "GetVisibleObjects",
                maxDistance=self.visible_distance,
                raise_for_failure=True,
            ).metadata["actionReturn"]
        )
        objects = env.step("GetObjectMetadata", objectIds=list(last_seen_object_ids)).metadata[
            "actionReturn"
        ]

        vis_gt = np.zeros(len(self.entity_types))
        for o in objects:
            if o["objectType"] in self.entity_types:
                idx = self.entity_types.index(o["objectType"])
                vis_gt[idx] = 1

        return vis_gt
    
    
class GoalObjectsVisibilitySensor(Sensor):
    def __init__(self, uuid, **kwargs):
        observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(len(TARGET_TYPES),),
            dtype=np.int64,
        )

        self.visible_distance = 5.0
        self.entity_types = TARGET_TYPES

        super().__init__(**prepare_locals_for_super(locals()))

        self.uuid = uuid

    def get_observation(self, env, task, *args: Any, **kwargs: Any) -> Any:
        last_seen_object_ids = set(
            env.step(
                "GetVisibleObjects",
                maxDistance=self.visible_distance,
                raise_for_failure=True,
            ).metadata["actionReturn"]
        )
        objects = env.step("GetObjectMetadata", objectIds=list(last_seen_object_ids)).metadata[
            "actionReturn"
        ]

        vis_gt = np.zeros(len(self.entity_types))
        for o in objects:
            if o["objectType"] in self.entity_types:
                idx = self.entity_types.index(o["objectType"])
                vis_gt[idx] = 1

        return vis_gt
    
    
class ValidMovesForwardSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = gym.spaces.Discrete(200)
        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(  # type:ignore
            self,
            env: IThorEnvironment,
            task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
            *args,
            **kwargs,
    ) -> np.ndarray:
        
        env_dict = env.__dict__.copy()
        
        agent_metadata = env.last_event.metadata['agent']
        pos = agent_metadata['position']
        rot = agent_metadata['rotation']
        hor = np.clip(agent_metadata['cameraHorizon'], -30., 30.)        
        
        valid_moves_forward = 0
        while env.step('MoveAhead').metadata['lastActionSuccess']:
            valid_moves_forward += 1

        env.step(
            action="TeleportFull",
            position=pos,
            rotation=rot,
            horizon=hor,
        )
        
        env.__dict__.update(env_dict)
        
        return valid_moves_forward
    
    
    
class FloorPercentageSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = gym.spaces.Dict(
            {
                "floor_percentage": gym.spaces.Box(
                    low=np.array([-np.inf], dtype=np.float32),
                    high=np.array([np.inf], dtype=np.float32),
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

        super().__init__(**prepare_locals_for_super(locals()))

    def get_observation(  # type:ignore
            self,
            env: IThorEnvironment,
            task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
            *args,
            **kwargs,
    ) -> np.ndarray:
        
        floor_colors = env.last_event.class_masks.instance_masks.class_colors["Floor"]
        image = env.last_event.semantic_segmentation_frame
        image_2d = image.reshape(-1, 3)
        color_mask = np.isin(image_2d, floor_colors).all(axis=1)
        count = np.count_nonzero(color_mask)

        # Calculate the percentage
        total_pixels = image.shape[0] * image.shape[1]
        percentage = (count / total_pixels)
        return percentage
    
class AggregateMapSensor(Sensor[Controller, Task[Controller]]):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(uuid, observation_space, **kwargs)

    def get_observation(
        self,
        env: Controller,
        task
    ) -> Any:
        return task.aggregate_map


class CurrentMapSensor(Sensor[Controller, Task[Controller]]):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = gym.spaces.Box(
            low=0, high=1, shape=(1,), dtype=np.float32
        )  # (low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        super().__init__(uuid, observation_space, **kwargs)

    def get_observation(
        self,
        env: Controller,
        task
    ) -> Any:
        return task.instant_map
    
    
class VisitedPosSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))
        

    def _get_observation_space(self) -> gym.spaces.Discrete:

        return gym.spaces.Discrete(2)

    def get_observation(  # type:ignore
            self,
            env: IThorEnvironment,
            task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
            *args,
            **kwargs,
    ) -> np.ndarray:
        
        position = env.last_event.metadata["agent"]["position"]
        if task.path is None or len(task.path) == 0: 
            return False
        return position in task.path[:-10]
    
    
class VisitedRoomSensor(
    Sensor[
        Union[RoboThorEnvironment, IThorEnvironment],
        Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
    ]
):
    def __init__(self, uuid: str, **kwargs: Any) -> None:
        observation_space = self._get_observation_space()

        super().__init__(**prepare_locals_for_super(locals()))
        

    def _get_observation_space(self) -> gym.spaces.Discrete:
        return gym.spaces.Discrete(2)

    def get_observation(  # type:ignore
            self,
            env: IThorEnvironment,
            task: Union[ObjectNaviThorGridTask, ObjectNavTask, PointNavTask],
            *args,
            **kwargs,
    ) -> np.ndarray:
        
        if task.current_room_id is None or task.ordered_visited_room_ids is None:
            return False
        
        return task.current_room_id in task.ordered_visited_room_ids[:-1]
    
    
class RewardWeightsSensor(Sensor):
    def __init__(self, uuid='reward_weights', num_objectives=5, **kwargs: Any):
        self.num_objectives = num_objectives
        observation_space = self._get_observation_space()
        super().__init__(**prepare_locals_for_super(locals()))

    def _get_observation_space(self):
        return gym.spaces.Discrete(self.num_objectives)

    def get_observation(self,env,task,*args: Any, **kwargs: Any) -> Any:
        config_idx = task.task_info['reward_weights']
        return np.array(config_idx)