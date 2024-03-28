import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union, Type, cast

from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.robothor_plugin.robothor_tasks import spl_metric

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import gym
import numpy as np
import torch
import time

import prior
import ai2thor
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering, Linux64
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.cache_utils import DynamicDistanceCache
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from procthor_objectnav.utils.map_utils import position_to_grid, reachable_positions_to_matrix
from allenact.utils.experiment_utils import set_seed
from procthor_objectnav.utils.weight_utils import random_weights, equally_spaced_weights

from shapely.geometry import Point, Polygon
from shapely.vectorized import contains
from procthor_objectnav.utils import (
    distance_to_object_id,
    distance_to_object_type,
    get_approx_geo_dist,
    get_room_connections,
    nearest_room_to_point,
    position_dist,
)
from procthor_objectnav.utils.types import (
    AgentPose,
    RewardConfig,
    TaskSamplerArgs,
    Vector3,
)

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

assets = ['Alarm_Clock_1', 'Alarm_Clock_12', 'Apple_1', 'Apple_10', 'BaseballBat_1', 'BaseballBat_2', \
          'Basketball_1', 'Bowl_1', 'Bowl_10', 'RoboTHOR_garbage_bin_ai2_1_v', 'bin_10', \
          'Houseplant_10', 'Houseplant_1', 'Laptop_1', 'Laptop_10', 'Mug_2', 'Mug_1', \
          'Spray_Bottle_1', 'Spray_Bottle_2', 'Vase_Decorative_1', 'Vase_Flat_1']

table_assets = ["Coffee_Table_201_1", "Dining_Table_16_1", "Desk_204_1", "RoboTHOR_side_table_havsta_v"]

indices = [558, 51, 203, 425, 735, 326, 176, 838, 108, 111, 553, 137, 461, 654, 386, 117, 610, 113, 748, 533, 554]


def spawn_tables(controller):
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]
    event = controller.step(
        action="SpawnAsset",
        assetId="Coffee_Table_201_1",
        generatedId=f"Coffee_Table_201_1",
        position=reachable_positions[101 % len(reachable_positions)],
        renderImage=False,
    )
    print("spawn table success: ", controller.last_event.metadata['lastActionSuccess'])
    event = controller.step(
        action="SpawnAsset",
        assetId="Dining_Table_16_1",
        generatedId=f"Dining_Table_16_1",
        position=reachable_positions[303 % len(reachable_positions)],
        renderImage=False,
    )
    print("spawn table success: ", controller.last_event.metadata['lastActionSuccess'])
    
    event = controller.step(
        action="SpawnAsset",
        assetId="Desk_204_1",
        generatedId=f"Desk_204_1",
        position=reachable_positions[505],
        renderImage=False,
    )
    print("spawn table success: ", controller.last_event.metadata['lastActionSuccess'])
    
    event = controller.step(
        action="SpawnAsset",
        assetId="RoboTHOR_side_table_havsta_v",
        generatedId=f"RoboTHOR_side_table_havsta_v",
        position=reachable_positions[707],
        renderImage=False,
    )
    print("spawn table success: ", controller.last_event.metadata['lastActionSuccess'])

def spawn_assets(controller):
    event = controller.step(action="GetReachablePositions")
    reachable_positions = event.metadata["actionReturn"]
    
    print("reachable positions: ", len(reachable_positions))

    for i, asset in enumerate(assets):
        event = controller.step(
                    action="SpawnAsset",
                    assetId=asset,
                    generatedId=asset,
                    position=reachable_positions[indices[i] % len(reachable_positions)],
                    renderImage=False,
            )

    controller.step(
            action="InitialRandomSpawn",
            randomSeed=123,
            forceVisible=True,
            numPlacementAttempts=10,
            placeStationary=True,
            numDuplicatesOfType=[{
                "objectType": type,
                "count": 2
            } for type in TARGET_TYPES],
            excludedReceptacles=[],
            excludedObjectIds=[])


class ProcTHORObjectNavTask(Task[Controller]):
    def __init__(
        self,
        controller: Controller,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_config: RewardConfig,
        distance_cache: DynamicDistanceCache,
        distance_type: Literal["geo", "l2", "approxGeo"] = "geo",
        visualize: Optional[bool] = None,
        house: Optional[Dict[str, Any]] = None,
        actions: Tuple[str] = ("MoveAhead", "RotateLeft", "RotateRight", "End", "LookUp", "LookDown"),
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            env=controller,
            sensors=sensors,
            task_info=task_info,
            max_steps=max_steps,
            **kwargs,
        )
        self.controller = controller
        self.house = house
        self.reward_config = reward_config
        self._took_end_action: bool = False
        self._success: Optional[bool] = False
        self.mirror = task_info["mirrored"]
        self.actions = actions

        self.seed: Optional[int] = None
        self.set_seed(seed)
        self.np_random = np.random.default_rng(self.seed)

        self._rewards: List[float] = []
        self._distance_to_goal: List[float] = []
        self._metrics = None
        self.path: List = (
            []
        )  # the initial coordinate will be directly taken from the optimal path
        self.travelled_distance = 0.0

        self.task_info["followed_path"] = [
            self.controller.last_event.metadata["agent"]["position"]
        ]
        self.task_info["taken_actions"] = []
        self.task_info["action_successes"] = []
        self.task_info["obs_embeds"] = []

        self.distance_cache = distance_cache

        self.distance_type = distance_type
        if distance_type == "geo":
            self.dist_to_target_func = self.min_geo_distance_to_target
        elif distance_type == "l2":
            self.dist_to_target_func = self.min_l2_distance_to_target
        elif distance_type == "approxGeo":
            self.dist_to_target_func = self.min_approx_geo_distance_to_target
            assert house is not None

            self.room_connection_graph, self.room_id_to_open_key = get_room_connections(
                house
            )

            self.room_polygons = [
                Polygon([(poly["x"], poly["z"]) for poly in room["floorPolygon"]])
                for room in house["rooms"]
            ]
        else:
            raise NotImplementedError

        self.last_distance = self.dist_to_target_func()
        self.optimal_distance = self.last_distance
        self.closest_distance = self.last_distance

        self.visualize = (
            visualize
            if visualize is not None
            else (self.task_info["mode"] == "eval" or random.random() < 1 / 1000)
        )
        self.observations = [self.controller.last_event.frame]
        self._metrics = None

        # save videos
        if "save_video" in kwargs and kwargs["save_video"]:
            self.save_video = True
            self.video_frames = {"rgb": [], "topdown": []}
            # event = self.controller.step(action="ToggleMapView")
            # cam_position = self.controller.last_event.metadata["cameraPosition"]
            # cam_rotation = {"x":90, "y":0, "z":0}
            # cam_orth_size = self.controller.last_event.metadata["cameraOrthSize"]
            # self.controller.step(
            #     action="AddThirdPartyCamera", skyboxColor="white", **{"position":cam_position, "rotation":cam_rotation, "orthographicSize":cam_orth_size, "orthographic":True}
            # )
        else:
            self.save_video = False
            
        # morl
        # reward shaping
        all_locations = [[k['x'], k['z']] for k in self.get_reachable_positions(self.controller)]
        self.all_reachable_positions = torch.Tensor(all_locations)
        self.has_visited = torch.zeros((len(self.all_reachable_positions), 1))

        # initial position
        current_agent_location = self.agent_state()
        current_agent_location = torch.Tensor([current_agent_location['x'], current_agent_location['z']])
        all_distances = self.all_reachable_positions - current_agent_location
        all_distances = (all_distances ** 2).sum(dim=-1)
        location_index = torch.argmin(all_distances)
        self.has_visited[location_index] = 1
        
        self.goal_observed_reward = False
        if "objectives" in kwargs:
            self.objectives = kwargs["objectives"]
            self.valid_objectives = kwargs["valid_objectives"]

            # buffer
            self._sub_rewards = {objective: [] for objective in self.objectives}

            if "safety" in self.objectives:
                self.min_x, self.max_x, self.min_y, self.max_y = self.all_reachable_positions[:, 0].min().item(), self.all_reachable_positions[:, 0].max().item(), self.all_reachable_positions[:, 1].min().item(), self.all_reachable_positions[:, 1].max().item()
                self.reachable_positions_matrix = reachable_positions_to_matrix(self.all_reachable_positions, grid_size=0.25)
                self.safety_grid = int(self.reward_config.safety_distance / 0.25)
                self.reachable_positions_matrix_shape = self.reachable_positions_matrix.shape

            if "object_exploration" in self.objectives:
                self.all_objects_visibility = {x['objectId']: x['visible'] for x in self.all_objects()}

        self.num_objectives = len(self.valid_objectives)
        self.adaptive_reward = kwargs["adaptive_reward"] if "adaptive_reward" in kwargs else False
        self.reward_embed_type = kwargs["reward_embed_type"] if "reward_embed_type" in kwargs else "codebook"
        self.validation_reward_weights = kwargs["validation_reward_weights"]
        self.test_reward_weights = kwargs["test_reward_weights"]
        if self.adaptive_reward:
            if self.reward_embed_type == "codebook" or self.reward_embed_type == "raw":
                self.reward_weights = random_weights(self.num_objectives, rng=self.np_random)
            elif self.reward_embed_type == "integer":
                weight_range = [0, 10]
                while True:
                    self.reward_weights = [np.random.choice(np.arange(weight_range[0], weight_range[1])) for _ in range(self.num_objectives)]
                    if sum(self.reward_weights) > 0:
                        break
                        
            if self.task_info["mode"] == "valid":
                self.reward_weights = list(self.validation_reward_weights[self.task_info["episode_index"]-1])
            elif self.task_info["mode"] == "test":
                self.reward_weights = list(self.test_reward_weights[self.task_info["episode_index"]-1])
        elif "reward_weights" in kwargs and kwargs["reward_weights"] is not None:
            self.reward_weights = kwargs["reward_weights"]
            
        self.normalized_reward_weights = np.array(self.reward_weights) / np.sum(self.reward_weights)
        self.task_info["reward_weights"] = list(self.reward_weights)

        # # debug
        # self.time_per_objectives = {objective: 0 for objective in self.objectives}

    def set_seed(self, seed: int):
        self.seed = seed
        if seed is not None:
            set_seed(seed)

    def agent_state(self, agent_id: int = 0) -> Dict:
        """Return agent position, rotation and horizon."""

        agent_meta = self.controller.last_event.metadata["agent"]
        return {
            **{k: float(v) for k, v in agent_meta["position"].items()},
            "rotation": {k: float(v) for k, v in agent_meta["rotation"].items()},
            "horizon": round(float(agent_meta["cameraHorizon"]), 1),
        }

    def all_objects(self) -> List[Dict[str, Any]]:
        """Return all object metadata."""
        return self.controller.last_event.metadata["objects"]

    def all_objects_with_properties(
        self, properties: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Find all objects with the given properties."""
        objects = []
        for o in self.all_objects():
            satisfies_all = True
            for k, v in properties.items():
                if o[k] != v:
                    satisfies_all = False
                    break
            if satisfies_all:
                objects.append(o)
        return objects

    def visible_objects(self) -> List[Dict[str, Any]]:
        """Return all visible objects."""
        return self.all_objects_with_properties({"visible": True})

    def min_approx_geo_distance_to_target(self) -> float:
        agent_position = self.controller.last_event.metadata["agent"]["position"]
        room_i_with_agent = nearest_room_to_point(
            point=agent_position, room_polygons=self.room_polygons
        )
        room_id_with_agent = int(
            self.house["rooms"][room_i_with_agent]["id"].split("|")[-1]
        )
        if room_id_with_agent in self.room_id_to_open_key:
            room_id_with_agent = self.room_id_to_open_key[room_id_with_agent]

        return get_approx_geo_dist(
            target_object_type=self.task_info["object_type"],
            agent_position=agent_position,
            house=self.house,
            controller=self.controller,
            room_polygons=self.room_polygons,
            room_connection_graph=self.room_connection_graph,
            room_id_to_open_key=self.room_id_to_open_key,
            room_id_with_agent=room_id_with_agent,
            house_name=self.task_info["house_name"],
        )

    def min_l2_distance_to_target(self) -> float:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = float("inf")
        obj_id_to_obj_pos = {
            o["objectId"]: o["axisAlignedBoundingBox"]["center"]
            for o in self.controller.last_event.metadata["objects"]
        }
        for object_id in self.task_info["target_object_ids"]:
            min_dist = min(
                min_dist,
                IThorEnvironment.position_dist(
                    obj_id_to_obj_pos[object_id],
                    self.controller.last_event.metadata["agent"]["position"],
                ),
            )
        if min_dist == float("inf"):
            get_logger().error(
                f"No target object {self.task_info['object_type']} found"
                f" in house {self.task_info['house_name']}."
            )
            return -1.0
        return min_dist

    def min_geo_distance_to_target(self) -> float:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        # NOTE: may return -1 if the object is unreachable.
        min_dist = None
        for object_id in self.task_info["target_object_ids"]:
            geo_dist = distance_to_object_id(
                controller=self.controller,
                distance_cache=self.distance_cache,
                object_id=object_id,
                house_name=self.task_info["house_name"],
            )
            if (min_dist is None and geo_dist >= 0) or (
                geo_dist >= 0 and geo_dist < min_dist
            ):
                min_dist = geo_dist
        if min_dist is None:
            return -1.0
        return min_dist

    @property
    def action_space(self):
        return gym.spaces.Discrete(len(self.actions))

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def class_action_names(self, **kwargs) -> Tuple[str, ...]:
        return self.actions

    def close(self) -> None:
        self.controller.stop()

    def _step(self, action: int) -> RLStepResult:
        if isinstance(action, List):
            action, obs_embeds = action[0], action[1]
            self.task_info["obs_embeds"].append(obs_embeds)
        else:
            assert isinstance(action, int)
            action = cast(int, action)

        action_str = self.class_action_names()[action]

        if self.mirror:
            if action_str == "RotateRight":
                action_str = "RotateLeft"
            elif action_str == "RotateLeft":
                action_str = "RotateRight"

        self.task_info["taken_actions"].append(action_str)

        if action_str == "End":
            self._took_end_action = True
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
            self.task_info["action_successes"].append(True)
        else:
            self.controller.step(action=action_str)
            self.last_action_success = bool(self.controller.last_event)

            position = self.controller.last_event.metadata["agent"]["position"]
            self.path.append(position)
            self.task_info["followed_path"].append(position)
            self.task_info["action_successes"].append(self.last_action_success)

        if self.save_video:
            rgb = self.controller.last_event.frame
            # topdown = self.controller.last_event.third_party_camera_frames[0]
            self.video_frames["rgb"].append(rgb)
            # self.video_frames["topdown"].append(topdown)

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            self.observations.append(self.controller.last_event.frame)

        sub_rewards, reward = self.judge()

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=reward,
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action, "sub_reward": sub_rewards} if not self.save_video\
                else {"last_action_success": self.last_action_success, "action": action, "sub_reward": sub_rewards, "rgb": self.video_frames["rgb"], "topdown": self.video_frames["topdown"]}
        )
        return step_result

    def render(
        self, mode: Literal["rgb", "depth"] = "rgb", *args, **kwargs
    ) -> np.ndarray:
        if mode == "rgb":
            frame = self.controller.last_event.frame.copy()
        elif mode == "depth":
            frame = self.controller.last_event.depth_frame.copy()
        else:
            raise NotImplementedError(f"Mode '{mode}' is not supported.")

        if self.mirror:
            frame = np.fliplr(frame)

        return frame

    def _is_goal_in_range(self) -> bool:
        return any(
            obj
            for obj in self.controller.last_event.metadata["objects"]
            if obj["visible"] and obj["objectType"] == self.task_info["object_type"]
        )

    def shaping(self) -> float:
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        cur_distance = self.dist_to_target_func()

        if self.distance_type == "l2":
            reward = max(self.closest_distance - cur_distance, 0)
            self.closest_distance = min(self.closest_distance, cur_distance)

            return reward * self.reward_config.shaping_weight
        else:
            # Ensuring the reward magnitude is not greater than the total distance moved
            max_reward_mag = 0.0
            if len(self.path) >= 2:
                p0, p1 = self.path[-2:]
                max_reward_mag = position_dist(p0, p1, ignore_y=True)

            if (
                self.reward_config.positive_only_reward
                and cur_distance is not None
                and cur_distance > 0.5
            ):
                reward = max(self.closest_distance - cur_distance, 0)
            elif self.last_distance is not None and cur_distance is not None:
                reward += self.last_distance - cur_distance

            self.last_distance = cur_distance
            self.closest_distance = min(self.closest_distance, cur_distance)

            return (
                max(
                    min(reward, max_reward_mag),
                    -max_reward_mag,
                )
                * self.reward_config.shaping_weight
            )

    def judge(self) -> float:
        """Judge the last event."""
        # default objectives
        default_objectives = [objective for objective in self.objectives if objective not in self.valid_objectives]

        # objectives = self.objectives
        objectives = default_objectives + [self.valid_objectives[idx] for idx in range(len(self.valid_objectives)) if self.reward_weights[idx] > 0]
        sub_rewards = self.judge_sub_rewards(objectives)
        reward = 0

        for idx in range(len(self.valid_objectives)):
            key = self.valid_objectives[idx]
            if self.normalized_reward_weights[idx] > 0:
                reward += sub_rewards[key] * self.normalized_reward_weights[idx]
            else:
                sub_rewards[key] = 0

        for objective in default_objectives:
            reward += sub_rewards[objective] * 1 # default rewards are not scaled

        self._rewards.append(float(reward))
        for objective in self.objectives:
            self._sub_rewards[objective].append(sub_rewards[objective])
        return sub_rewards, float(reward)

    def judge_sub_rewards(self, objectives=["step_penalty", "path_efficiency", "goal"]) -> float:
        """Judge the last event."""
        sub_reward_dict = {}
        sub_reward_dict["step_penalty"] = self.reward_config.step_penalty
        sub_reward_dict["path_efficiency"] = self.shaping()

        if "exploration" in objectives or "safety" in objectives:
            current_agent_state = self.agent_state()
            current_agent_location = torch.Tensor([current_agent_state['x'], current_agent_state['z']])
        
        if "exploration" in objectives:
            all_distances = self.all_reachable_positions - current_agent_location
            all_distances = (all_distances ** 2).sum(dim=-1)
            location_index = torch.argmin(all_distances)
            if self.has_visited[location_index] == 0:
                visited_new_place = True
            else:
                visited_new_place = False
            self.has_visited[location_index] = 1

            if visited_new_place and not self.goal_observed_reward:
                sub_reward_dict["exploration"] = self.reward_config.exploration_reward
            else:
                sub_reward_dict["exploration"] = 0.0

        if "safety" in objectives:
            agent_grid = position_to_grid(current_agent_state, min_x=self.min_x, min_y=self.min_y, grid_size=0.25)
            
            min_x_safety_grid = max(0, agent_grid[0]-self.safety_grid)
            max_x_safety_grid = min(self.reachable_positions_matrix_shape[0], agent_grid[0]+self.safety_grid+1)
            min_y_safety_grid = max(0, agent_grid[1]-self.safety_grid)
            max_y_safety_grid = min(self.reachable_positions_matrix_shape[1], agent_grid[1]+self.safety_grid+1)
            
            safety_reward = np.sum(self.reachable_positions_matrix[min_x_safety_grid:max_x_safety_grid, min_y_safety_grid:max_y_safety_grid])

            if safety_reward > self.reward_config.safety_reward_threshold:
                perception_distance_threshold = 2.0
                visible_target_object_list = [visible_object for visible_object in self.controller.last_event.metadata['objects'] if visible_object['objectType']==self.task_info['object_type'] and visible_object['distance']<perception_distance_threshold]
                goal_is_visible = (len(visible_target_object_list) > 0)

                if not goal_is_visible:
                    sub_reward_dict["safety"] = - safety_reward * self.reward_config.safety_reward_scale
                else:
                    sub_reward_dict["safety"] = 0.0
            else:
                sub_reward_dict["safety"] = 0.0

        if "object_exploration" in objectives:
            new_visible = 0

            for object in self.visible_objects():
                if object['objectId'] not in self.all_objects_visibility:
                    self.all_objects_visibility[object['objectId']] = True
                    new_visible += 1
                elif not self.all_objects_visibility[object['objectId']]:
                    self.all_objects_visibility[object['objectId']] = True
                    new_visible += 1

            sub_reward_dict["object_exploration"] = self.reward_config.object_found * new_visible / len(self.all_objects_visibility)


        if "goal" in objectives:
            if self._took_end_action:
                if self._success:
                    sub_reward_dict["goal"] = self.reward_config.goal_success_reward
                else:
                    sub_reward_dict["goal"] = self.reward_config.failed_stop_reward
            else:
                sub_reward_dict["goal"] = 0.0

        return sub_reward_dict

    def get_reachable_positions(self, controller):
        event = controller.step('GetReachablePositions')
        reachable_positions = event.metadata['actionReturn']
        if reachable_positions is None or len(reachable_positions) == 0:
            print('Scene name', controller.last_event.metadata['sceneName'])
        return reachable_positions

    def get_observations(self, **kwargs) -> Any:
        obs = super().get_observations()
        if self.mirror:
            for o in obs:
                if ("rgb" in o or "depth" in o) and isinstance(obs[o], np.ndarray):
                    obs[o] = np.fliplr(obs[o])
        return obs

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        metrics["dist_to_target"] = self.dist_to_target_func()
        metrics["total_reward"] = np.sum(self._rewards)
        metrics["spl"] = spl_metric(
            success=self._success,
            optimal_distance=self.optimal_distance,
            travelled_distance=self.travelled_distance,
        )
        metrics["spl"] = 0.0 if metrics["spl"] is None else metrics["spl"]
        # sub_reward metrics
        metrics.update({"sub_reward/total_{}".format(objective): np.sum(self._sub_rewards[objective]) for objective in self.objectives})
        metrics["sub_rewards"] = self._sub_rewards
        metrics["success"] = self._success

        self._metrics = metrics
        return metrics


class ProcTHORObjectNavToInstanceTask(ProcTHORObjectNavTask):
    def __init__(
        self,
        controller: Controller,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_config: RewardConfig,
        distance_cache: DynamicDistanceCache,
        distance_type: str = "geo",
        visualize: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))
        self.distance_cache = None
        assert self.distance_type == "l2"

        self.plausible_target_ids = set(self.task_info["target_object_ids"])
        self.seen_plausible_target_ids = set()
        self.task_info["target_object_ids"] = [
            random.choice(self.task_info["target_object_ids"])
        ]
        self.target_object_id = self.task_info["target_object_ids"][0]

        self._num_newly_seen_targets = 0

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def _step(self, action: int) -> RLStepResult:
        action_str = self.class_action_names()[action]

        if self.mirror:
            if action_str == "RotateRight":
                action_str = "RotateLeft"
            elif action_str == "RotateLeft":
                action_str = "RotateRight"

        self.task_info["taken_actions"].append(action_str)

        self._num_newly_seen_targets = 0
        if action_str == "End":
            visible_object_ids = set(
                o["objectId"]
                for o in self.controller.last_event.metadata["objects"]
                if o["visible"]
            )
            self._took_end_action = (
                self.target_object_id in visible_object_ids
                or len(visible_object_ids & self.plausible_target_ids) == 0
            )
            self._num_newly_seen_targets = len(
                (visible_object_ids & self.plausible_target_ids)
                - self.seen_plausible_target_ids
            )
            self.seen_plausible_target_ids |= (
                visible_object_ids & self.plausible_target_ids
            )
            self._success = self._is_goal_in_range()
            self.last_action_success = self._success
            self.task_info["action_successes"].append(True)
        else:
            self.controller.step(action=action_str)
            self.last_action_success = bool(self.controller.last_event)

            position = self.controller.last_event.metadata["agent"]["position"]
            self.path.append(position)
            self.task_info["followed_path"].append(position)
            self.task_info["action_successes"].append(self.last_action_success)

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            self.observations.append(self.controller.last_event.frame)

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def _is_goal_in_range(self) -> bool:
        return any(
            obj
            for obj in self.controller.last_event.metadata["objects"]
            if obj["visible"] and obj["objectId"] == self.target_object_id
        )

    def _unseen_target_in_range(self) -> bool:
        return (
            len(
                set(
                    obj["objectId"]
                    for obj in self.controller.last_event.metadata["objects"]
                    if obj["visible"] and obj["objectId"] in self.plausible_target_ids
                )
                - self.seen_plausible_target_ids
            )
            != 0
        )

    def shaping(self) -> float:
        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        cur_distance = self.dist_to_target_func()

        if self.distance_type == "l2":
            reward = max(self.closest_distance - cur_distance, 0)
            self.closest_distance = min(self.closest_distance, cur_distance)

            return reward * self.reward_config.shaping_weight
        else:
            # Ensuring the reward magnitude is not greater than the total distance moved
            max_reward_mag = 0.0
            if len(self.path) >= 2:
                p0, p1 = self.path[-2:]
                max_reward_mag = position_dist(p0, p1, ignore_y=True)

            if (
                self.reward_config.positive_only_reward
                and cur_distance is not None
                and cur_distance > 0.5
            ):
                reward = max(self.closest_distance - cur_distance, 0)
            elif self.last_distance is not None and cur_distance is not None:
                reward += self.last_distance - cur_distance

            self.last_distance = cur_distance
            self.closest_distance = min(self.closest_distance, cur_distance)

            return (
                max(
                    min(reward, max_reward_mag),
                    -max_reward_mag,
                )
                * self.reward_config.shaping_weight
            )

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_config.step_penalty

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config.goal_success_reward
            else:
                reward += self.reward_config.failed_stop_reward
        elif self._num_newly_seen_targets != 0:
            reward += (
                self.reward_config.goal_success_reward
                # * self._num_newly_seen_targets
            )
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        metrics["seen_plausible_targets"] = len(self.seen_plausible_target_ids)
        metrics["saw_other_target_before_success"] = 1 * (
            metrics["success"] and len(self.seen_plausible_target_ids) > 1
        )
        metrics["found_object_of_type"] = 1 * (len(self.seen_plausible_target_ids) > 0)
        return metrics


class ProcTHORObjectNavToAllInstancesTask(ProcTHORObjectNavTask):
    def __init__(
        self,
        controller: Controller,
        sensors: List[Sensor],
        task_info: Dict[str, Any],
        max_steps: int,
        reward_config: RewardConfig,
        distance_cache: DynamicDistanceCache,
        distance_type: str = "geo",
        visualize: Optional[bool] = None,
        **kwargs,
    ) -> None:
        super().__init__(**prepare_locals_for_super(locals()))
        self.distance_cache = None
        assert self.distance_type == "l2"

        self.plausible_target_ids = set(self.task_info["target_object_ids"])
        self.seen_plausible_target_ids = set()

        self._num_newly_seen_targets = 0

        self.object_id_to_closest_distance = self.l2_distance_to_remaining_targets()

    def min_l2_distance_to_target(self) -> float:
        # TODO: VERY HACKY, making this into a max L2 function across targets
        #   as this is what should be used for this task
        dists = list(self.l2_distance_to_remaining_targets().values())
        if len(dists) == 0:
            return 0.0
        return max(dists)

    def reached_terminal_state(self) -> bool:
        return self._took_end_action

    def _step(self, action: int) -> RLStepResult:
        action_str = self.class_action_names()[action]

        if self.mirror:
            if action_str == "RotateRight":
                action_str = "RotateLeft"
            elif action_str == "RotateLeft":
                action_str = "RotateRight"

        self.task_info["taken_actions"].append(action_str)

        self._num_newly_seen_targets = 0
        if action_str == "End":
            visible_object_ids = set(
                o["objectId"]
                for o in self.controller.last_event.metadata["objects"]
                if o["visible"]
            )
            self._num_newly_seen_targets = len(
                (visible_object_ids & self.plausible_target_ids)
                - self.seen_plausible_target_ids
            )
            self.seen_plausible_target_ids |= (
                visible_object_ids & self.plausible_target_ids
            )

            self.task_info["target_object_ids"] = list(
                self.plausible_target_ids - self.seen_plausible_target_ids
            )
            self._success = len(self.task_info["target_object_ids"]) == 0
            self._took_end_action = (
                self._success
                or len(visible_object_ids & self.plausible_target_ids) == 0
            )

            self.last_action_success = self._success
            self.task_info["action_successes"].append(True)
        else:
            self.controller.step(action=action_str)
            self.last_action_success = bool(self.controller.last_event)

            position = self.controller.last_event.metadata["agent"]["position"]
            self.path.append(position)
            self.task_info["followed_path"].append(position)
            self.task_info["action_successes"].append(self.last_action_success)

        if len(self.path) > 1:
            self.travelled_distance += position_dist(
                p0=self.path[-1], p1=self.path[-2], ignore_y=True
            )

        if self.visualize:
            self.observations.append(self.controller.last_event.frame)

        step_result = RLStepResult(
            observation=self.get_observations(),
            reward=self.judge(),
            done=self.is_done(),
            info={"last_action_success": self.last_action_success, "action": action},
        )
        return step_result

    def _is_goal_in_range(self) -> bool:
        raise NotImplementedError

    def _unseen_target_in_range(self) -> bool:
        return (
            len(
                set(
                    obj["objectId"]
                    for obj in self.controller.last_event.metadata["objects"]
                    if obj["visible"] and obj["objectId"] in self.plausible_target_ids
                )
                - self.seen_plausible_target_ids
            )
            != 0
        )

    def l2_distance_to_remaining_targets(self) -> Dict[str, float]:
        """Return the minimum distance to a target object.

        May return a negative value if the target object is not reachable.
        """
        obj_id_to_obj_pos = {
            o["objectId"]: o["axisAlignedBoundingBox"]["center"]
            for o in self.controller.last_event.metadata["objects"]
        }
        object_id_to_dist = {}
        for object_id in self.task_info["target_object_ids"]:
            object_id_to_dist[object_id] = IThorEnvironment.position_dist(
                obj_id_to_obj_pos[object_id],
                self.controller.last_event.metadata["agent"]["position"],
            )
        return object_id_to_dist

    def shaping(self) -> float:
        assert self.distance_type == "l2"

        if self.reward_config.shaping_weight == 0.0:
            return 0

        reward = 0.0
        cur_dists = self.l2_distance_to_remaining_targets()
        if self._num_newly_seen_targets == 0:
            assert len(cur_dists) != 0
            reward = max(
                max(self.object_id_to_closest_distance[oid] - cur_dists[oid], 0)
                for oid in self.object_id_to_closest_distance
            )
            reward = reward * self.reward_config.shaping_weight

        try:
            self.object_id_to_closest_distance = {
                oid: min(self.object_id_to_closest_distance[oid], cur_dists[oid])
                for oid in cur_dists  # Must be `for oid in cur_dists` as we want to throw away oid's that may have been seen
            }
        except:
            print(
                self.object_id_to_closest_distance,
                cur_dists,
                self.seen_plausible_target_ids,
                self.plausible_target_ids,
                self.task_info["target_object_ids"],
            )
        self.last_distance = None

        return reward

    def judge(self) -> float:
        """Judge the last event."""
        reward = self.reward_config.step_penalty

        reward += self.shaping()

        if self._took_end_action:
            if self._success:
                reward += self.reward_config.goal_success_reward
            else:
                reward += self.reward_config.failed_stop_reward
        elif self._num_newly_seen_targets != 0:
            reward += (
                self.reward_config.goal_success_reward
                # * self._num_newly_seen_targets
            )
        elif self.num_steps_taken() + 1 >= self.max_steps:
            reward += self.reward_config.reached_horizon_reward

        self._rewards.append(float(reward))
        return float(reward)

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        metrics["seen_plausible_targets"] = len(self.seen_plausible_target_ids)
        metrics["prop_targets_seen"] = len(self.seen_plausible_target_ids) / len(
            self.plausible_target_ids
        )
        metrics["saw_other_target_before_success"] = 1 * (
            metrics["success"] and len(self.seen_plausible_target_ids) > 1
        )
        metrics["found_object_of_type"] = 1 * (len(self.seen_plausible_target_ids) > 0)
        return metrics

# class ProcTHORObjectNavTaskSamplerEval(TaskSampler):
#     def __init__(
#         self,
#         args: TaskSamplerArgs,
#         object_nav_task_type: Type = ProcTHORObjectNavTask,
#         **extra_task_kwargs,
#     ) -> None:
#         self.args = args
#         self.extra_task_kwargs = extra_task_kwargs
#         self.object_nav_task_type = object_nav_task_type
#         random.shuffle(self.args.house_inds)

#         self.controller: Optional[Controller] = None
#         self.distance_cache = DynamicDistanceCache(rounding=1)

#         # get the total number of tasks assigned to this process
#         self.reset_tasks = self.args.max_tasks

#         self.resample_same_scene_freq = args.resample_same_scene_freq
#         """The number of times to resample the same houses before moving to the next on."""

#         self.house_inds_index = 0
#         self.episode_index = 0

#         self._last_sampled_task: Optional[ProcTHORObjectNavTask] = None

#         self.reachable_positions_map: Dict[int, Vector3] = dict()
#         self.objects_in_scene_map: Dict[str, List[str]] = dict()

#         self.visible_objects_cache = dict()

#         rotate_step_degrees = self.args.controller_args["rotateStepDegrees"]
#         self.valid_rotations = np.arange(
#             start=0, stop=360, step=rotate_step_degrees
#         ).tolist()

#         if args.seed is not None:
#             self.set_seed(args.seed)

#         if args.deterministic_cudnn:
#             set_deterministic_cudnn()

#         self.target_object_types_set = set(self.args.target_object_types)
#         self.obj_type_counter = Counter(
#             {obj_type: 0 for obj_type in self.args.target_object_types}
#         )

#         self.reset()

#     def set_seed(self, seed: int):
#         set_seed(seed)

#     @property
#     def length(self) -> Union[int, float]:
#         """Length.
#         # Returns
#         Number of total tasks remaining that can be sampled. Can be float('inf').
#         """
#         return self.args.max_tasks

#     @property
#     def total_unique(self) -> Optional[Union[int, float]]:
#         return self.reset_tasks

#     @property
#     def last_sampled_task(self) -> Optional[ProcTHORObjectNavTask]:
#         # NOTE: This book-keeping should be done in TaskSampler...
#         return self._last_sampled_task

#     def close(self) -> None:
#         if self.controller is not None:
#             self.controller.stop()

#     @property
#     def all_observation_spaces_equal(self) -> bool:
#         """Check if observation spaces equal.
#         # Returns
#         True if all Tasks that can be sampled by this sampler have the
#             same observation space. Otherwise False.
#         """
#         return True

#     def get_nearest_positions(self, world_position: Vector3) -> List[Vector3]:
#         """Get the n reachable positions that are closest to the world_position."""
#         self.reachable_positions.sort(
#             key=lambda p: sum((p[k] - world_position[k]) ** 2 for k in ["x", "z"])
#         )
#         return self.reachable_positions[
#             : min(
#                 len(self.reachable_positions),
#                 self.args.max_agent_positions,
#             )
#         ]

#     def get_nearest_agent_height(self, y_coordinate: float) -> float:
#         """Get the nearest valid agent height to a y_coordinate."""
#         if len(self.args.valid_agent_heights) == 1:
#             return self.args.valid_agent_heights[0]

#         min_distance = float("inf")
#         out = None
#         for height in self.args.valid_agent_heights:
#             dist = abs(y_coordinate - height)
#             if dist < min_distance:
#                 min_distance = dist
#                 out = height
#         return out

#     @property
#     def house_index(self) -> int:
#         return self.args.house_inds[self.house_inds_index]

#     def is_object_visible(self, object_id: str) -> bool:
#         """Return True if object_id is visible without any interaction in the scene.

#         This method makes an approximation based on checking if the object
#         is hit with a raycast from nearby reachable positions.
#         """
#         # NOTE: Check the cached visible objects first.
#         if (
#             self.house_index in self.visible_objects_cache
#             and object_id in self.visible_objects_cache[self.house_index]
#         ):
#             return self.visible_objects_cache[self.house_index][object_id]
#         elif self.house_index not in self.visible_objects_cache:
#             self.visible_objects_cache[self.house_index] = dict()

#         # NOTE: Get the visibility points on the object
#         visibility_points = self.controller.step(
#             action="GetVisibilityPoints", objectId=object_id, raise_for_failure=True
#         ).metadata["actionReturn"]

#         # NOTE: Randomly sample visibility points
#         for vis_point in random.sample(
#             population=visibility_points,
#             k=min(len(visibility_points), self.args.max_vis_points),
#         ):
#             # NOTE: Get the nearest reachable agent positions to the target object.
#             agent_positions = self.get_nearest_positions(world_position=vis_point)
#             for agent_pos in agent_positions:
#                 agent_pos = agent_pos.copy()
#                 agent_pos["y"] = self.get_nearest_agent_height(
#                     y_coordinate=vis_point["y"]
#                 )
#                 event = self.controller.step(
#                     action="PerformRaycast",
#                     origin=agent_pos,
#                     destination=vis_point,
#                 )
#                 hit = event.metadata["actionReturn"]
#                 if (
#                     event.metadata["lastActionSuccess"]
#                     and hit["objectId"] == object_id
#                     and hit["hitDistance"] < self.args.controller_args["visibilityDistance"]
#                 ):
#                     self.visible_objects_cache[self.house_index][object_id] = True
#                     return True

#         self.visible_objects_cache[self.house_index][object_id] = False
#         return False

#     @property
#     def target_objects_in_scene(self) -> Dict[str, List[str]]:
#         """Return a map from the object type to the objectIds in the scene."""
#         if self.house_index in self.objects_in_scene_map:
#             return self.objects_in_scene_map[self.house_index]

#         event = self.controller.step(action="ResetObjectFilter", raise_for_failure=True)
#         objects = event.metadata["objects"]
#         out = {}
#         for obj in objects:
#             if obj["objectType"] in self.target_object_types_set:
#                 if obj["objectType"] not in out:
#                     out[obj["objectType"]] = []
#                 out[obj["objectType"]].append(obj["objectId"])
#         self.objects_in_scene_map[self.house_index] = out
#         return out

#     def sample_target_object_ids(self) -> Tuple[str, List[str]]:
#         """Sample target objects.

#         Objects returned will all be of the same objectType. Only considers visible
#         objects in the house.
#         """
#         if random.random() < self.args.p_greedy_target_object:
#             for obj_type, count in reversed(self.obj_type_counter.most_common()):
#                 instances_of_type = self.target_objects_in_scene.get(obj_type, [])

#                 # NOTE: object type doesn't appear in the scene.
#                 if not instances_of_type:
#                     continue

#                 visible_ids = []
#                 for object_id in instances_of_type:
#                     if self.is_object_visible(object_id=object_id):
#                         visible_ids.append(object_id)

#                 if visible_ids:
#                     self.obj_type_counter[obj_type] += 1
#                     return obj_type, visible_ids
#         else:
#             candidates = dict()
#             for obj_type, object_ids in self.target_objects_in_scene.items():
#                 visible_ids = []
#                 for object_id in object_ids:
#                     if self.is_object_visible(object_id=object_id):
#                         visible_ids.append(object_id)

#                 if visible_ids:
#                     candidates[obj_type] = visible_ids

#             if candidates:
#                 return random.choice(list(candidates.items()))

#         raise ValueError(f"No target objects in house {self.house_index}.")

#     @property
#     def reachable_positions(self) -> List[Vector3]:
#         """Return the reachable positions in the current house."""
#         return self.reachable_positions_map[self.house_index]

#     def increment_scene(self) -> bool:
#         """Increment the current scene.

#         Returns True if the scene works with reachable positions, False otherwise.
#         """
#         self.increment_scene_index()

#         # self.controller.step(action="DestroyHouse", raise_for_failure=True)
#         self.controller.reset()
#         self.house = self.args.houses[self.house_index]

#         self.controller.step(
#             action="CreateHouse", house=self.house, raise_for_failure=True
#         )
        
#         print('Spawning assets!!')
#         spawn_assets(self.controller)
#         print('Spawning tables!!')
#         spawn_tables(self.controller)

#         # NOTE: Set reachable positions
#         if self.house_index not in self.reachable_positions_map:
#             pose = self.house["metadata"]["agent"].copy()
#             if self.args.controller_args["agentMode"] == "locobot":
#                 del pose["standing"]
#             event = self.controller.step(action="TeleportFull", **pose)
#             if not event:
#                 get_logger().warning(f"Initial teleport failing in {self.house_index}.")
#                 return False
#             rp_event = self.controller.step(action="GetReachablePositions")
#             if not rp_event:
#                 # NOTE: Skip scenes where GetReachablePositions fails
#                 get_logger().warning(
#                     f"GetReachablePositions failed in {self.house_index}"
#                 )
#                 return False
#             reachable_positions = rp_event.metadata["actionReturn"]
#             self.reachable_positions_map[self.house_index] = reachable_positions
#         return True

#     def increment_scene_index(self):
#         self.house_inds_index = (self.house_inds_index + 1) % len(self.args.house_inds)

#     def next_task(
#         self, force_advance_scene: bool = False
#     ) -> Optional[ProcTHORObjectNavTask]:
#         # NOTE: Stopping condition
#         if self.args.max_tasks <= 0:
#             return None

#         # NOTE: Setup the Controller
#         if self.controller is None:

#             if self.args.controller_args.get("commit_id", "") not in ["", None]:
#                 if "branch" in self.args.controller_args:
#                     del self.args.controller_args["branch"]

#             self.controller = Controller(**self.args.controller_args)
#             get_logger().info(
#                 f"Using Controller commit id: {self.controller._build.commit_id}"
#             )
#             while not self.increment_scene():
#                 pass
            
            

#         # NOTE: determine if the house should be changed.
#         # if (
#         #     force_advance_scene
#         #     or (
#         #         self.resample_same_scene_freq > 0
#         #         and self.episode_index % self.resample_same_scene_freq == 0
#         #     )
#         #     or self.episode_index == 0
#         # ):
#         #     while not self.increment_scene():
#         #         pass
            

#         # NOTE: Choose target object
#         while True:
#             try:
#                 # NOTE: The loop avoid a very rare edge case where the agent
#                 # starts out trapped in some part of the room.
#                 target_object_type, target_object_ids = self.sample_target_object_ids()
#                 break
#             except ValueError:
#                 while not self.increment_scene():
#                     pass
                

#         self.controller.step(action="ResetMaterials", raise_for_failure=True)

#         door_ids = [door["id"] for door in self.house["doors"]]
#         self.controller.step(
#             action="SetObjectFilter",
#             objectIds=target_object_ids + door_ids,
#             raise_for_failure=True,
#         )

#         # NOTE: Set agent pose
#         standing = (
#             {}
#             if self.args.controller_args["agentMode"] == "locobot"
#             else {"standing": True}
#         )
#         starting_pose = AgentPose(
#             position=random.choice(self.reachable_positions),
#             rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
#             horizon=30,
#             **standing,
#         )
#         event = self.controller.step(action="TeleportFull", **starting_pose)
#         if not event:
#             get_logger().warning(
#                 f"Teleport failing in {self.house_index} at {starting_pose}"
#             )

#         self.episode_index += 1
#         self.args.max_tasks -= 1

#         task_kwargs = dict(
#             controller=self.controller,
#             sensors=self.args.sensors,
#             max_steps=self.args.max_steps,
#             reward_config=self.args.reward_config,
#             distance_type=self.args.distance_type,
#             house=self.house,
#             distance_cache=self.distance_cache,
#             actions=self.args.actions,
#             task_info={
#                 "mode": self.args.mode,
#                 "process_ind": self.args.process_ind,
#                 "house_name": str(self.house_index),
#                 "rooms": len(self.house["rooms"]),
#                 "target_object_ids": target_object_ids,
#                 "object_type": target_object_type,
#                 "starting_pose": starting_pose,
#                 "mirrored": self.args.allow_flipping and random.random() > 0.5,
#             },
#             # morl
#             save_video=self.args.save_video,
#             save_trajectory=self.args.save_trajectory,
#             # objectives_combination=self.args.objectives_combination,
#             objectives=self.args.objectives,
#             valid_objectives=self.args.valid_objectives,
#             adaptive_reward=self.args.adaptive_reward,
#             reward_weights=self.args.reward_weights,
#         )
#         task_kwargs.update(self.extra_task_kwargs)
#         self._last_sampled_task = self.object_nav_task_type(**task_kwargs)
#         return self._last_sampled_task

#     def reset(self):
#         self.episode_index = 0
#         self.args.max_tasks = self.reset_tasks
#         self.house_inds_index = 0


class ProcTHORObjectNavTaskSampler(TaskSampler):
    def __init__(
        self,
        args: TaskSamplerArgs,
        object_nav_task_type: Type = ProcTHORObjectNavTask,
        **extra_task_kwargs,
    ) -> None:
        self.args = args
        self.extra_task_kwargs = extra_task_kwargs
        self.object_nav_task_type = object_nav_task_type
        random.shuffle(self.args.house_inds)

        self.controller: Optional[Controller] = None
        self.distance_cache = DynamicDistanceCache(rounding=1)

        # get the total number of tasks assigned to this process
        self.reset_tasks = self.args.max_tasks

        self.resample_same_scene_freq = args.resample_same_scene_freq
        """The number of times to resample the same houses before moving to the next on."""

        self.house_inds_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[ProcTHORObjectNavTask] = None

        self.reachable_positions_map: Dict[int, Vector3] = dict()
        self.objects_in_scene_map: Dict[str, List[str]] = dict()

        self.visible_objects_cache = dict()

        rotate_step_degrees = self.args.controller_args["rotateStepDegrees"]
        self.valid_rotations = np.arange(
            start=0, stop=360, step=rotate_step_degrees
        ).tolist()

        if args.seed is not None:
            self.set_seed(args.seed)

        if args.deterministic_cudnn:
            set_deterministic_cudnn()

        self.target_object_types_set = set(self.args.target_object_types)
        self.obj_type_counter = Counter(
            {obj_type: 0 for obj_type in self.args.target_object_types}
        )

        self.reset()

        self.reset_eval_weights()

    def set_seed(self, seed: int):
        set_seed(seed)

    def reset_eval_weights(self):
        self.num_objectives = len(self.args.valid_objectives)
        num_eval_weights_for_front = 1000
        # if self.args.mode == "train" or self.args.mode == "valid":
        #     self.validation_reward_weights = equally_spaced_weights(self.num_objectives, n=num_eval_weights_for_front)
        #     self.test_reward_weights = None
        # elif self.args.mode == "test":
        #     self.validation_reward_weights = None
        #     self.test_reward_weights = equally_spaced_weights(self.num_objectives, n=num_eval_weights_for_front)
        self.validation_reward_weights = equally_spaced_weights(self.num_objectives, n=num_eval_weights_for_front)
        self.test_reward_weights = equally_spaced_weights(self.num_objectives, n=num_eval_weights_for_front)


    @property
    def length(self) -> Union[int, float]:
        """Length.
        # Returns
        Number of total tasks remaining that can be sampled. Can be float('inf').
        """
        return self.args.max_tasks

    @property
    def total_unique(self) -> Optional[Union[int, float]]:
        return self.reset_tasks

    @property
    def last_sampled_task(self) -> Optional[ProcTHORObjectNavTask]:
        # NOTE: This book-keeping should be done in TaskSampler...
        return self._last_sampled_task

    def close(self) -> None:
        if self.controller is not None:
            self.controller.stop()

    @property
    def all_observation_spaces_equal(self) -> bool:
        """Check if observation spaces equal.
        # Returns
        True if all Tasks that can be sampled by this sampler have the
            same observation space. Otherwise False.
        """
        return True

    def get_nearest_positions(self, world_position: Vector3) -> List[Vector3]:
        """Get the n reachable positions that are closest to the world_position."""
        self.reachable_positions.sort(
            key=lambda p: sum((p[k] - world_position[k]) ** 2 for k in ["x", "z"])
        )
        return self.reachable_positions[
            : min(
                len(self.reachable_positions),
                self.args.max_agent_positions,
            )
        ]

    def get_nearest_agent_height(self, y_coordinate: float) -> float:
        """Get the nearest valid agent height to a y_coordinate."""
        if len(self.args.valid_agent_heights) == 1:
            return self.args.valid_agent_heights[0]

        min_distance = float("inf")
        out = None
        for height in self.args.valid_agent_heights:
            dist = abs(y_coordinate - height)
            if dist < min_distance:
                min_distance = dist
                out = height
        return out

    @property
    def house_index(self) -> int:
        return self.args.house_inds[self.house_inds_index]

    def is_object_visible(self, object_id: str) -> bool:
        """Return True if object_id is visible without any interaction in the scene.

        This method makes an approximation based on checking if the object
        is hit with a raycast from nearby reachable positions.
        """
        # NOTE: Check the cached visible objects first.
        if (
            self.house_index in self.visible_objects_cache
            and object_id in self.visible_objects_cache[self.house_index]
        ):
            return self.visible_objects_cache[self.house_index][object_id]
        elif self.house_index not in self.visible_objects_cache:
            self.visible_objects_cache[self.house_index] = dict()

        # NOTE: Get the visibility points on the object
        visibility_points = self.controller.step(
            action="GetVisibilityPoints", objectId=object_id, raise_for_failure=True
        ).metadata["actionReturn"]

        # NOTE: Randomly sample visibility points
        for vis_point in random.sample(
            population=visibility_points,
            k=min(len(visibility_points), self.args.max_vis_points),
        ):
            # NOTE: Get the nearest reachable agent positions to the target object.
            agent_positions = self.get_nearest_positions(world_position=vis_point)
            for agent_pos in agent_positions:
                agent_pos = agent_pos.copy()
                agent_pos["y"] = self.get_nearest_agent_height(
                    y_coordinate=vis_point["y"]
                )
                event = self.controller.step(
                    action="PerformRaycast",
                    origin=agent_pos,
                    destination=vis_point,
                )
                hit = event.metadata["actionReturn"]
                if (
                    event.metadata["lastActionSuccess"]
                    and hit["objectId"] == object_id
                    and hit["hitDistance"] < self.args.controller_args["visibilityDistance"]
                ):
                    self.visible_objects_cache[self.house_index][object_id] = True
                    return True

        self.visible_objects_cache[self.house_index][object_id] = False
        return False

    @property
    def target_objects_in_scene(self) -> Dict[str, List[str]]:
        """Return a map from the object type to the objectIds in the scene."""
        if self.house_index in self.objects_in_scene_map:
            return self.objects_in_scene_map[self.house_index]

        event = self.controller.step(action="ResetObjectFilter", raise_for_failure=True)
        objects = event.metadata["objects"]
        out = {}
        for obj in objects:
            if obj["objectType"] in self.target_object_types_set:
                if obj["objectType"] not in out:
                    out[obj["objectType"]] = []
                out[obj["objectType"]].append(obj["objectId"])
        self.objects_in_scene_map[self.house_index] = out
        return out

    def sample_target_object_ids(self) -> Tuple[str, List[str]]:
        """Sample target objects.

        Objects returned will all be of the same objectType. Only considers visible
        objects in the house.
        """
        if random.random() < self.args.p_greedy_target_object:
            for obj_type, count in reversed(self.obj_type_counter.most_common()):
                instances_of_type = self.target_objects_in_scene.get(obj_type, [])

                # NOTE: object type doesn't appear in the scene.
                if not instances_of_type:
                    continue

                visible_ids = []
                for object_id in instances_of_type:
                    if self.is_object_visible(object_id=object_id):
                        visible_ids.append(object_id)

                if visible_ids:
                    self.obj_type_counter[obj_type] += 1
                    return obj_type, visible_ids
        else:
            candidates = dict()
            for obj_type, object_ids in self.target_objects_in_scene.items():
                visible_ids = []
                for object_id in object_ids:
                    if self.is_object_visible(object_id=object_id):
                        visible_ids.append(object_id)

                if visible_ids:
                    candidates[obj_type] = visible_ids

            if candidates:
                return random.choice(list(candidates.items()))

        raise ValueError(f"No target objects in house {self.house_index}.")

    @property
    def reachable_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.reachable_positions_map[self.house_index]

    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.increment_scene_index()

        # self.controller.step(action="DestroyHouse", raise_for_failure=True)
        self.controller.reset()
        self.house = self.args.houses[self.house_index]

        self.controller.step(
            action="CreateHouse", house=self.house, raise_for_failure=True
        )

        # NOTE: Set reachable positions
        if self.house_index not in self.reachable_positions_map:
            pose = self.house["metadata"]["agent"].copy()
            if self.args.controller_args["agentMode"] == "locobot":
                del pose["standing"]
            event = self.controller.step(action="TeleportFull", **pose)
            if not event:
                get_logger().warning(f"Initial teleport failing in {self.house_index}.")
                return False
            rp_event = self.controller.step(action="GetReachablePositions")
            if not rp_event:
                # NOTE: Skip scenes where GetReachablePositions fails
                get_logger().warning(
                    f"GetReachablePositions failed in {self.house_index}"
                )
                return False
            reachable_positions = rp_event.metadata["actionReturn"]
            self.reachable_positions_map[self.house_index] = reachable_positions
        return True

    def increment_scene_index(self):
        self.house_inds_index = (self.house_inds_index + 1) % len(self.args.house_inds)

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None
        
        try:

            # NOTE: Setup the Controller
            if self.controller is None:

                if self.args.controller_args.get("commit_id", "") not in ["", None]:
                    if "branch" in self.args.controller_args:
                        del self.args.controller_args["branch"]

                self.controller = Controller(**self.args.controller_args)
                get_logger().info(
                    f"Using Controller commit id: {self.controller._build.commit_id}"
                )
                while not self.increment_scene():
                    pass

            # NOTE: determine if the house should be changed.
            if (
                force_advance_scene
                or (
                    self.resample_same_scene_freq > 0
                    and self.episode_index % self.resample_same_scene_freq == 0
                )
                or self.episode_index == 0
            ):
                while not self.increment_scene():
                    pass
                
            # NOTE: Choose target object
            while True:
                try:
                    # NOTE: The loop avoid a very rare edge case where the agent
                    # starts out trapped in some part of the room.
                    target_object_type, target_object_ids = self.sample_target_object_ids()
                    break
                except ValueError:
                    while not self.increment_scene():
                        pass
                    

            if random.random() < self.args.p_randomize_materials:
                self.controller.step(action="RandomizeMaterials", raise_for_failure=True)
            else:
                self.controller.step(action="ResetMaterials", raise_for_failure=True)

            door_ids = [door["id"] for door in self.house["doors"]]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids + door_ids,
                raise_for_failure=True,
            )

            # NOTE: Set agent pose
            standing = (
                {}
                if self.args.controller_args["agentMode"] == "locobot"
                else {"standing": True}
            )
            def get_random_pose():
                try:
                    starting_pose = AgentPose(
                        position=random.choice(self.reachable_positions),
                        rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
                        horizon=30,
                        **standing,
                    )
                    event = self.controller.step(action="TeleportFull", **starting_pose)
                except:
                    print('Failed to teleport! Trying again ...')
                    return get_random_pose()
                return starting_pose, event

            starting_pose, event = get_random_pose()
                
            if not event:
                get_logger().warning(
                    f"Teleport failing in {self.house_index} at {starting_pose}"
                )
        except:
            print('Failed to sample next task! Trying again ...')
            return self.next_task(force_advance_scene=force_advance_scene)

        self.episode_index += 1
        self.args.max_tasks -= 1

        # if self.validation_reward_weights does not exist as a class variable, run self.reset_eval_weights()
        if not hasattr(self, "validation_reward_weights"):
            self.reset_eval_weights()

        task_kwargs = dict(
            controller=self.controller,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type=self.args.distance_type,
            house=self.house,
            distance_cache=self.distance_cache,
            actions=self.args.actions,
            task_info={
                "mode": self.args.mode,
                "process_ind": self.args.process_ind,
                "house_name": str(self.house_index),
                "rooms": len(self.house["rooms"]),
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
                "episode_index": self.episode_index,
            },
            # morl
            save_video=self.args.save_video,
            save_trajectory=self.args.save_trajectory,
            # objectives_combination=self.args.objectives_combination,
            objectives=self.args.objectives,
            valid_objectives=self.args.valid_objectives,
            adaptive_reward=self.args.adaptive_reward,
            reward_weights=self.args.reward_weights,
            validation_reward_weights=self.validation_reward_weights,
            test_reward_weights=self.test_reward_weights,
            reward_embed_type=self.args.reward_embed_type,
        )
        task_kwargs.update(self.extra_task_kwargs)
        self._last_sampled_task = self.object_nav_task_type(**task_kwargs)
        return self._last_sampled_task

    def reset(self):
        self.episode_index = 0
        self.args.max_tasks = self.reset_tasks
        self.house_inds_index = 0

class FullProcTHORObjectNavHideObjectsTestTaskSampler(ProcTHORObjectNavTaskSampler):
    """Works with PRIOR's object-nav-eval tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=min(self.reset_tasks // 10, 4), dtype=np.uint8
            ).tolist()
        )
        self.args.controller_args = self.args.controller_args.copy()
        self.houses = prior.load_dataset("procthor-10k")[
            "val" if self.args.test_on_validation or not (self.args.mode == "eval") else "test"
        ]
        
    def hide_objects(
        self,
        obj_types: List = []
    ):
        object_ids_to_hide = [
            obj["objectId"]
            for obj in self.controller.last_event.metadata["objects"]
            if obj["objectType"] in obj_types
        ]
        for obj_id in object_ids_to_hide:
            print(f"Hiding object {obj_id}")
            self.controller.step(
                action="DisableObject",
                objectId=obj_id
            )
            

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_scene is None or self.last_scene != ep["scene"]:
                if isinstance(ep["scene"], str) and "ArchitecTHOR" in ep["scene"] and "_" in ep["scene"]:
                    ep["scene"] = ep["scene"].replace("_", "-")
                    n = int(ep["scene"][-1])
                    ep["scene"] = "{}{}".format(ep["scene"][:-1], n - 1)
                self.last_scene = ep["scene"]
                self.controller.reset(
                    scene=(
                        self.houses[ep["scene"]]
                        if ep["sceneDataset"] == "procthor-10k"
                        else ep["scene"]
                    )
                )
                
            self.hide_objects(['DiningTable', 'CoffeeTable', 'SideTable', 'Desk',  'TVStand'])

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ProcTHORObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                actions=self.args.actions,
                task_info={
                    "mode": self.args.mode,
                    "house_name": str(ep["scene"]),
                    "sceneDataset": ep["sceneDataset"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task

class FullProcTHORObjectNavSpawnObjectsTestTaskSampler(ProcTHORObjectNavTaskSampler):
    """Works with PRIOR's object-nav-eval tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=min(self.reset_tasks // 10, 4), dtype=np.uint8
            ).tolist()
        )
        self.args.controller_args = self.args.controller_args.copy()
        self.houses = prior.load_dataset("procthor-10k")[
            "val" if self.args.test_on_validation or not (self.args.mode == "eval") else "test"
        ]
            

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_scene is None or self.last_scene != ep["scene"]:
                if isinstance(ep["scene"], str) and "ArchitecTHOR" in ep["scene"] and "_" in ep["scene"]:
                    ep["scene"] = ep["scene"].replace("_", "-")
                    n = int(ep["scene"][-1])
                    ep["scene"] = "{}{}".format(ep["scene"][:-1], n - 1)
                self.last_scene = ep["scene"]
                self.controller.reset(
                    scene=(
                        self.houses[ep["scene"]]
                        if ep["sceneDataset"] == "procthor-10k"
                        else ep["scene"]
                    )
                )
                
            print('Spawning tomatoes!!')
                
            # get reachable positions
            event = self.controller.step(action="GetReachablePositions")
            reachable_positions = event.metadata["actionReturn"]

            for i in range(1):
                event = self.controller.step(
                    action="SpawnAsset",
                    assetId="Tomato_1",
                    generatedId=f"Tomato_{i}",
                    position=random.choice(reachable_positions),
                    renderImage=False,
                )
            # self.controller.step(
            #     action="InitialRandomSpawn",
            #     randomSeed=123,
            #     forceVisible=False,
            #     numPlacementAttempts=10,
            #     placeStationary=True,
            #     numDuplicatesOfType=[{
            #         "objectType": "Tomato",
            #         "count": 200
            #     }],
            #     excludedReceptacles=[],
            #     excludedObjectIds=[])
                

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ProcTHORObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                actions=self.args.actions,
                task_info={
                    "mode": self.args.mode,
                    "house_name": str(ep["scene"]),
                    "sceneDataset": ep["sceneDataset"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task

class FullProcTHORObjectNavCreateObjectsAtLocationTestTaskSampler(ProcTHORObjectNavTaskSampler):
    """Works with PRIOR's object-nav-eval tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=min(self.reset_tasks // 10, 4), dtype=np.uint8
            ).tolist()
        )
        self.args.controller_args = self.args.controller_args.copy()
        self.houses = prior.load_dataset("procthor-10k")[
            "val" if self.args.test_on_validation or not (self.args.mode == "eval") else "test"
        ]
            

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_scene is None or self.last_scene != ep["scene"]:
                if isinstance(ep["scene"], str) and "ArchitecTHOR" in ep["scene"] and "_" in ep["scene"]:
                    ep["scene"] = ep["scene"].replace("_", "-")
                    n = int(ep["scene"][-1])
                    ep["scene"] = "{}{}".format(ep["scene"][:-1], n - 1)
                self.last_scene = ep["scene"]
                self.controller.reset(
                    scene=(
                        self.houses[ep["scene"]]
                        if ep["sceneDataset"] == "procthor-10k"
                        else ep["scene"]
                    )
                )
                
            # Create Objects at Reachable Positions
            
            # get reachable positions
            event = self.controller.step(action="GetReachablePositions")
            reachable_positions = event.metadata["actionReturn"]

            # valid rotations
            valid_rotations = np.arange(
                start=0, stop=360, step=30
                ).tolist()

            # create object at reachable position
            print("Creating tomato!!!")
            self.controller.step(
                action="CreateObjectAtLocation",
                objectType="Tomato",
                position=random.choice(reachable_positions),
                rotation=Vector3(x=30, y=30, z=30),
            )

                

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ProcTHORObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                actions=self.args.actions,
                task_info={
                    "mode": self.args.mode,
                    "house_name": str(ep["scene"]),
                    "sceneDataset": ep["sceneDataset"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task



class FullProcTHORObjectNavTestTaskSampler(ProcTHORObjectNavTaskSampler):
    """Works with PRIOR's object-nav-eval tasks."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=min(self.reset_tasks // 10, 4), dtype=np.uint8
            ).tolist()
        )
        self.args.controller_args = self.args.controller_args.copy()
        self.houses = prior.load_dataset("procthor-10k")[
            "val" if self.args.test_on_validation or not (self.args.mode == "eval") else "test"
        ]

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_scene is None or self.last_scene != ep["scene"]:
                if isinstance(ep["scene"], str) and "ArchitecTHOR" in ep["scene"] and "_" in ep["scene"]:
                    ep["scene"] = ep["scene"].replace("_", "-")
                    n = int(ep["scene"][-1])
                    ep["scene"] = "{}{}".format(ep["scene"][:-1], n - 1)
                self.last_scene = ep["scene"]
                self.controller.reset(
                    scene=(
                        self.houses[ep["scene"]]
                        if ep["sceneDataset"] == "procthor-10k"
                        else ep["scene"]
                    )
                )

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ProcTHORObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                actions=self.args.actions,
                task_info={
                    "mode": self.args.mode,
                    "house_name": str(ep["scene"]),
                    "sceneDataset": ep["sceneDataset"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
                # morl
                save_video=self.args.save_video,
                save_trajectory=self.args.save_trajectory,
                # objectives_combination=self.args.objectives_combination,
                objectives=self.args.objectives,
                valid_objectives=self.args.valid_objectives,
                adaptive_reward=self.args.adaptive_reward,
                reward_weights=self.args.reward_weights,
                validation_reward_weights=self.validation_reward_weights,
                test_reward_weights=self.test_reward_weights,
                reward_embed_type=self.args.reward_embed_type,
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task


class ProcTHORObjectNavTestTaskSampler(ProcTHORObjectNavTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=min(self.reset_tasks // 10, 4), dtype=np.uint8
            ).tolist()
        )
        self.args.controller_args = self.args.controller_args.copy()

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_scene is None or self.last_scene != ep["scene"]:
                self.last_scene = ep["scene"]
                self.controller.reset(ep["scene"])

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ProcTHORObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                actions=self.args.actions,
                task_info={
                    "mode": self.args.mode,
                    "house_name": ep["scene"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task


class ProcTHORObjectNavInDomainTestTaskSampler(ProcTHORObjectNavTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scene = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=min(self.reset_tasks // 10, 4), dtype=np.uint8
            ).tolist()
        )
        self.args.controller_args = self.args.controller_args.copy()

        houses = prior.load_dataset("procthor-10k")
        if (self.args.mode == "eval") and not self.args.test_on_validation:
            self.houses = houses["test"]
        else:
            self.houses = houses["validation"]

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_scene is None or self.last_scene != ep["scene"]:
                self.last_scene = ep["scene"]
                self.controller.reset(scene="Procedural")
                house = self.houses[ep["scene"]]
                self.controller.step(action="CreateHouse", house=house)

            # NOTE: not using ep["targetObjectIds"] due to floating points with
            # target objects moving.
            event = self.controller.step(action="ResetObjectFilter")
            target_object_ids = [
                obj["objectId"]
                for obj in event.metadata["objects"]
                if obj["objectType"] == ep["targetObjectType"]
            ]
            self.controller.step(
                action="SetObjectFilter",
                objectIds=target_object_ids,
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                # NOTE: Skip scenes where TeleportFull fails.
                # This is added from a bug in the RoboTHOR eval dataset.
                get_logger().error(
                    f"Teleport failing {event.metadata['actionReturn']} in {epidx}."
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ProcTHORObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                actions=self.args.actions,
                task_info={
                    "mode": self.args.mode,
                    "house_name": ep["scene"],
                    "target_object_ids": target_object_ids,
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['scene']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task


class ProcTHORProcTHORObjectNavTestTaskSampler(ProcTHORObjectNavTaskSampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_house_idx = None

        # visualize 1/10 episodes
        self.epids_to_visualize = set(
            np.linspace(
                0, self.reset_tasks, num=self.reset_tasks // 10, dtype=np.uint8
            ).tolist()
        )

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        while True:
            # NOTE: Stopping condition
            if self.args.max_tasks <= 0:
                return None

            # NOTE: Setup the Controller
            if self.controller is None:
                self.controller = Controller(**self.args.controller_args)

            epidx = self.args.house_inds[self.args.max_tasks - 1]
            ep = self.args.houses[epidx]

            if self.last_house_idx is None or self.last_house_idx != ep["house"]:
                # NOTE: reusing unused variable name. Had trouble adding one real quick.
                houses_dataset = self.args.resample_same_scene_freq

                house = houses_dataset[ep["house"]]
                self.controller.reset()
                self.controller.step(
                    action="CreateHouse", house=house, raise_for_failure=True
                )
                self.last_house_idx = ep["house"]

            self.controller.step(
                action="SetObjectFilter",
                objectIds=ep["targetObjectIds"],
                raise_for_failure=True,
            )

            event = self.controller.step(action="TeleportFull", **ep["agentPose"])
            if not event:
                get_logger().error(
                    f"Teleport failing in {ep['house']} at {ep['agentPose']}"
                )
                self.args.max_tasks -= 1
                self.episode_index += 1
                continue

            difficulty = {"difficulty": ep["difficulty"]} if "difficulty" in ep else {}
            self._last_sampled_task = ProcTHORObjectNavTask(
                visualize=self.episode_index in self.epids_to_visualize,
                controller=self.controller,
                sensors=self.args.sensors,
                max_steps=self.args.max_steps,
                reward_config=self.args.reward_config,
                distance_type=self.args.distance_type,
                distance_cache=self.distance_cache,
                actions=self.args.actions,
                task_info={
                    "mode": self.args.mode,
                    "house_name": ep["house"],
                    "target_object_ids": ep["targetObjectIds"],
                    "object_type": ep["targetObjectType"],
                    "starting_pose": ep["agentPose"],
                    "mirrored": False,
                    "id": f"{ep['house']}__proc{self.args.process_ind}__global{epidx}__{ep['targetObjectType']}",
                    **difficulty,
                },
            )

            self.args.max_tasks -= 1
            self.episode_index += 1

            return self._last_sampled_task


class TrainiTHORTaskSamplerProcTHOR(ProcTHORObjectNavTaskSampler):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        random.shuffle(self.args.houses)

    @property
    def reachable_positions(self) -> List[Vector3]:
        """Return the reachable positions in the current house."""
        return self.reachable_positions_map[self.house_inds_index]

    @property
    def house_index(self) -> int:
        return self.house_inds_index

    def increment_scene(self) -> bool:
        """Increment the current scene.

        Returns True if the scene works with reachable positions, False otherwise.
        """
        self.house_inds_index = (self.house_inds_index + 1) % len(self.args.houses)
        self.controller.reset(scene=self.args.houses[self.house_inds_index])
        event = self.controller.step(action="GetReachablePositions")
        if not event:
            return False
        self.reachable_positions_map[self.house_inds_index] = event.metadata[
            "actionReturn"
        ]
        return True

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

        # NOTE: Setup the Controller
        if self.controller is None:
            self.controller = Controller(**self.args.controller_args)
            get_logger().info(
                f"Using Controller commit id: {self.controller._build.commit_id}"
            )

        # NOTE: determine if the house should be changed.
        if force_advance_scene or (
            self.resample_same_scene_freq > 0
            and self.episode_index % self.resample_same_scene_freq == 0
        ):
            while not self.increment_scene():
                pass

        shuffled_objects = False
        if self.args.ithor_p_shuffle_objects > 0:
            # NOTE: Don't cache visible objects because they might change.
            self.visible_objects_cache = dict()

            # NOTE: Must update because objectIds update after initial random spawn.
            self.objects_in_scene_map = dict()

            if random.random() < self.args.ithor_p_shuffle_objects:
                event = self.controller.step(
                    action="InitialRandomSpawn",
                    randomSeed=random.randint(0, 2**30),
                    forceVisible=True,
                )
                if event:
                    shuffled_objects = True
                else:
                    get_logger().warning("InitialRandomSpawn failing!")

        # NOTE: Choose target object
        while True:
            try:
                # NOTE: The loop avoid a very rare edge case where the agent
                # starts out trapped in some part of the room.
                target_object_type, target_object_ids = self.sample_target_object_ids()
                break
            except ValueError:
                while not self.increment_scene():
                    pass

        if random.random() < self.args.p_randomize_materials:
            self.controller.step(action="RandomizeMaterials", raise_for_failure=True)
        else:
            self.controller.step(action="ResetMaterials", raise_for_failure=True)

        self.controller.step(
            action="SetObjectFilter",
            objectIds=target_object_ids,
            raise_for_failure=True,
        )

        # NOTE: Set agent pose
        standing = (
            {}
            if self.args.controller_args["agentMode"] == "locobot"
            else {"standing": True}
        )
        starting_pose = AgentPose(
            position=random.choice(self.reachable_positions),
            rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
            horizon=30,
            **standing,
        )
        event = self.controller.step(action="TeleportFull", **starting_pose)
        if not event:
            get_logger().warning(
                f"Teleport failing in {self.house_index} at {starting_pose}"
            )

        self.episode_index += 1
        self.args.max_tasks -= 1

        self._last_sampled_task = ProcTHORObjectNavTask(
            controller=self.controller,
            sensors=self.args.sensors,
            max_steps=self.args.max_steps,
            reward_config=self.args.reward_config,
            distance_type=self.args.distance_type,
            distance_cache=self.distance_cache,
            actions=self.args.actions,
            task_info={
                "mode": self.args.mode,
                "process_ind": self.args.process_ind,
                "rooms": 1,
                "house_name": self.args.houses[self.house_inds_index],
                "target_object_ids": target_object_ids,
                "object_type": target_object_type,
                "starting_pose": starting_pose,
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
                "shuffled_objects": shuffled_objects,
            },
        )
        return self._last_sampled_task