import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union, Type
from collections import deque

from allenact.utils.misc_utils import prepare_locals_for_super
from allenact_plugins.robothor_plugin.robothor_tasks import spl_metric

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import gym
import numpy as np
import prior
import ai2thor
import copy
import json
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering, Linux64
from allenact.base_abstractions.misc import RLStepResult
from allenact.base_abstractions.sensor import Sensor
from allenact.base_abstractions.task import Task, TaskSampler
from allenact.utils.cache_utils import DynamicDistanceCache
from allenact.utils.experiment_utils import set_deterministic_cudnn, set_seed
from allenact.utils.system import get_logger
from allenact_plugins.ithor_plugin.ithor_environment import IThorEnvironment
from shapely.geometry import Point, Polygon
from shapely.vectorized import contains
from utils.map_utils import build_centered_map, update_aggregate_map_blocky


from procthor_objectnav.tasks.object_nav import ProcTHORObjectNavTask, ProcTHORObjectNavTaskSampler
from procthor_objectnav.tasks.flee_nav import ProcTHORFleeNavTask, ProcTHORFleeNavTaskSampler
from procthor_objectnav.utils import (
    distance_to_object_id,
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
from procthor_objectnav.tasks.constants import random_inds

unrewarded_objects = ["Floor","Wall","Doorway","Doorframe","Window","ShelvingUnit","CounterTop","Shelf","Drawer"]


def rooms_visited(path, room_polymap:dict, previously_visited_rooms = {}):
    elimination_polymap = {k: v for k, v in room_polymap.items() if k not in previously_visited_rooms}
    visited_rooms = copy.deepcopy(previously_visited_rooms)
    current_room_id = None
    for agent_pose in path:
        for room_id, poly in visited_rooms.items():
            if poly.contains(Point(agent_pose['x'],agent_pose['z'])):
                current_room_id = room_id
                break

        for room_id, poly in elimination_polymap.items():
            if poly.contains(Point(agent_pose['x'],agent_pose['z'])):
                del elimination_polymap[room_id]
                visited_rooms[room_id] = poly
                current_room_id = room_id
                break
    return visited_rooms, previously_visited_rooms, current_room_id

def get_rooms_polymap_and_type(house):
    room_poly_map = {}
    room_type_dict = {}
    for i, room in enumerate(house["rooms"]):
        room_poly_map[room["id"]] = Polygon(
            [(p["x"], p["z"]) for p in room["floorPolygon"]]
        )
        room_type_dict[room["id"]] = room['roomType']
    return room_poly_map, room_type_dict


class ProcTHORObjectNavMappingTask(ProcTHORObjectNavTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregate_map = None
        self.instant_map = None
        self.room_polys = None
        self.seen_object_ids = set()
        self.visited_rooms = None
        self.object_counter = None
        self.consecutive_move_aheads = [0]
        self.num_visited_pos = 0
        self.current_room_id = None
        self.ordered_visited_room_ids = []


    def set_map(self,base_map, room_polys):
        self.aggregate_map = copy.deepcopy(base_map["map"])
        self.map_scaling_params = {key: base_map[key] for key in ["xyminmax","pixel_sizes"]}
        m,n,_ = self.aggregate_map.shape
        # self.instant_map = np.concatenate((copy.deepcopy(base_map["map"]),np.zeros([m,n,11])),axis=2)
        self.room_polys = room_polys
        self.map_observations = [self.sensor_suite.sensors['aggregate_map_sensor'].get_observation(self.controller,self)]
    
    def set_counter(self,object_counter):
        self.object_counter = object_counter

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        # number_rooms_visited = len(self.visited_rooms)
        # metrics['percentage_rooms_visited'] = number_rooms_visited/len(self.room_polys)
        # metrics['room_visitation_efficiency'] = number_rooms_visited/metrics["ep_length"]
        # metrics['map_coverage'] = np.sum(self.aggregate_map[:,:,[2]].flatten())/(np.sum(self.aggregate_map[:,:,[0]].flatten()))
        # metrics['map_exploration_efficiency'] = np.sum(self.aggregate_map[:,:,[2]].flatten())/metrics["ep_length"]
        # metrics['seen_objects'] = len(self.seen_object_ids)
        # metrics['new_object_rate'] = len(self.seen_object_ids)/metrics["ep_length"]
        # all_object_types = ['Sofa','Television','Bed','Chair','Toilet','AlarmClock',
        #                     'Apple','BaseballBat','BasketBall','Bowl','GarbageCan',
        #                     'HousePlant','Laptop','Mug','SprayBottle','Vase']
        # for obj in all_object_types:
        #     metrics['z_object_fraction_'+obj] = 0
        #     metrics['z_counter_'+obj] = self.object_counter[obj]
        #     if sum(self.object_counter.values())>0:
        #         metrics['z_counter_fraction_'+obj] = self.object_counter[obj]/sum(self.object_counter.values())
        #     else:
        #         metrics['z_counter_fraction_'+obj]=0
        # metrics['z_object_fraction_'+self.task_info["object_type"]] = 1

        # if not self._success:
        #     metrics['map_coverage_failure'] = metrics['map_coverage']
        #     metrics['map_exp_eff_failure'] = metrics['map_exploration_efficiency']
        #     metrics['fail_eplen'] = metrics["ep_length"]
        #     metrics['seen_objects_fail'] = len(self.seen_object_ids)
            
        # else:
        #     # metrics['map_coverage_success'] = metrics['map_coverage']
        #     metrics['map_exp_eff_sucess'] = metrics['map_exploration_efficiency']
        #     metrics['sucess_eplen'] = metrics["ep_length"]
        #     metrics['seen_objects_sucess'] = len(self.seen_object_ids)
            
        # metrics['consecutive_move_aheads'] = sum(self.consecutive_move_aheads) / len(self.consecutive_move_aheads)
        # metrics['visited_pos'] = self.num_visited_pos / metrics["ep_length"]
        
        
        self._metrics = metrics
        return metrics
    
    def _step(self, action: int) -> RLStepResult:
        sr = super()._step(action)
        # current_pose = self.controller.last_event.metadata["agent"]
        # heading_idx = int(np.round(current_pose["rotation"]["y"] / 30)) % 11
        # possible_headings = [0.0]*12 
        # possible_headings[heading_idx] = 1
        # map_update = [current_pose['position']['x'],current_pose['position']['z'],1]
        # self.aggregate_map = update_aggregate_map_blocky(self.aggregate_map,[map_update],**self.map_scaling_params)
        # # self.instant_map = map_utils.populate_instantaneous_map(self.instant_map,[map_update+possible_headings])

        # ## New objects
        # visible_object_ids = set(
        #     o["objectId"]
        #     for o in self.controller.last_event.metadata["objects"]
        #     if o["visible"]
        #     and o["objectType"] not in unrewarded_objects
        # )

        
        # self.seen_object_ids.update(visible_object_ids)
        
        # # consecutive move aheads
        # if action == 1:
        #     self.consecutive_move_aheads[-1] += 1
        # else:
        #     self.consecutive_move_aheads.append(0)
            
        # # visited pose
        # if self.controller.last_event.metadata["agent"]["position"] in self.path[:-1]:
        #     self.num_visited_pos += 1
        
        
        # ## New rooms
        # if self.visited_rooms is None:
        #     self.visited_rooms, _, self.current_room_id = rooms_visited(self.path,self.room_polys)
        # else:
        #     self.visited_rooms,previously_visited_rooms, self.current_room_id = rooms_visited(self.path[-3:],self.room_polys,self.visited_rooms)
        # if not self.ordered_visited_room_ids or self.ordered_visited_room_ids[-1] != self.current_room_id:
        #     self.ordered_visited_room_ids.append(self.current_room_id)
            

        # ## Map video
        # if self.visualize:
        #     self.map_observations.append(self.sensor_suite.sensors['aggregate_map_sensor'].get_observation(self.controller,self))

        return sr
    
    
class ProcTHORObjectNavTaskSamplerEval(ProcTHORObjectNavTaskSampler):
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
        self.counter = 0

    def set_seed(self, seed: int):
        set_seed(seed)
        
        
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
                # return random.choice(list(candidates.items()))
                return list(candidates.items())[random_inds[self.counter] % len(candidates.items())]

        raise ValueError(f"No target objects in house {self.house_index}.")

    

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORObjectNavTask]:
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

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
        # starting_pose = AgentPose(
        #     position=random.choice(self.reachable_positions),
        #     rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
        #     horizon=30,
        #     **standing,
        # )
        starting_pose = AgentPose(
            position=self.reachable_positions[random_inds[self.counter] % len(self.reachable_positions)],
            rotation=Vector3(x=0, y=self.valid_rotations[random_inds[self.counter] % len(self.valid_rotations)], z=0),
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
            },
            # morl
            save_video=self.args.save_video,
            save_trajectory=self.args.save_trajectory,
            # objectives_combination=self.args.objectives_combination,
            objectives=self.args.objectives,
            valid_objectives=self.args.valid_objectives,
            adaptive_reward=self.args.adaptive_reward,
            reward_weights=self.args.reward_weights,
        )
        task_kwargs.update(self.extra_task_kwargs)
        self._last_sampled_task = self.object_nav_task_type(**task_kwargs)
        self.counter = (self.counter + 1) % 1000
        return self._last_sampled_task


    
class ProcTHORObjectNavTaskMappingSampler(ProcTHORObjectNavTaskSamplerEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_cache = dict()
        self.room_poly_cache = dict()
    
    def get_house_map(self):
        if (self.house_index in self.map_cache):
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]
        else:
            base_map, xyminmax, pixel_sizes = build_centered_map(self.house)
            self.map_cache[self.house_index] = {"map":base_map,"xyminmax":xyminmax, "pixel_sizes":pixel_sizes}
            self.room_poly_cache[self.house_index],_ = get_rooms_polymap_and_type(self.house)
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ProcTHORObjectNavMappingTask]:
        parent_task = super().next_task(force_advance_scene)
        if parent_task is not None:
            # parent_task.set_map(*self.get_house_map())
            parent_task.set_counter(self.obj_type_counter)
        self._last_sampled_task = parent_task
        return self._last_sampled_task
    
    

class ProcTHORFleeNavMappingTask(ProcTHORFleeNavTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.aggregate_map = None
        self.instant_map = None
        self.room_polys = None
        self.seen_object_ids = set()
        self.visited_rooms = None
        self.object_counter = None
        self.consecutive_move_aheads = [0]
        self.num_visited_pos = 0
        self.current_room_id = None
        self.ordered_visited_room_ids = []


    def set_map(self,base_map, room_polys):
        self.aggregate_map = copy.deepcopy(base_map["map"])
        self.map_scaling_params = {key: base_map[key] for key in ["xyminmax","pixel_sizes"]}
        m,n,_ = self.aggregate_map.shape
        # self.instant_map = np.concatenate((copy.deepcopy(base_map["map"]),np.zeros([m,n,11])),axis=2)
        self.room_polys = room_polys
        self.map_observations = [self.sensor_suite.sensors['aggregate_map_sensor'].get_observation(self.controller,self)]
    
    def set_counter(self,object_counter):
        self.object_counter = object_counter

    def metrics(self) -> Dict[str, Any]:
        if not self.is_done():
            return {}

        metrics = super().metrics()
        # number_rooms_visited = len(self.visited_rooms)
        # metrics['percentage_rooms_visited'] = number_rooms_visited/len(self.room_polys)
        # metrics['room_visitation_efficiency'] = number_rooms_visited/metrics["ep_length"]
        # metrics['map_coverage'] = np.sum(self.aggregate_map[:,:,[2]].flatten())/(np.sum(self.aggregate_map[:,:,[0]].flatten()))
        # metrics['map_exploration_efficiency'] = np.sum(self.aggregate_map[:,:,[2]].flatten())/metrics["ep_length"]
        # metrics['seen_objects'] = len(self.seen_object_ids)
        # metrics['new_object_rate'] = len(self.seen_object_ids)/metrics["ep_length"]
        # all_object_types = ['Sofa','Television','Bed','Chair','Toilet','AlarmClock',
        #                     'Apple','BaseballBat','BasketBall','Bowl','GarbageCan',
        #                     'HousePlant','Laptop','Mug','SprayBottle','Vase']
        # for obj in all_object_types:
        #     metrics['z_object_fraction_'+obj] = 0
        #     metrics['z_counter_'+obj] = self.object_counter[obj]
        #     if sum(self.object_counter.values())>0:
        #         metrics['z_counter_fraction_'+obj] = self.object_counter[obj]/sum(self.object_counter.values())
        #     else:
        #         metrics['z_counter_fraction_'+obj]=0
        # metrics['z_object_fraction_'+self.task_info["object_type"]] = 1

        # if not self._success:
        #     metrics['map_coverage_failure'] = metrics['map_coverage']
        #     metrics['map_exp_eff_failure'] = metrics['map_exploration_efficiency']
        #     metrics['fail_eplen'] = metrics["ep_length"]
        #     metrics['seen_objects_fail'] = len(self.seen_object_ids)
            
        # else:
        #     # metrics['map_coverage_success'] = metrics['map_coverage']
        #     metrics['map_exp_eff_sucess'] = metrics['map_exploration_efficiency']
        #     metrics['sucess_eplen'] = metrics["ep_length"]
        #     metrics['seen_objects_sucess'] = len(self.seen_object_ids)
            
        # metrics['consecutive_move_aheads'] = sum(self.consecutive_move_aheads) / len(self.consecutive_move_aheads)
        # metrics['visited_pos'] = self.num_visited_pos / metrics["ep_length"]
        
        
        self._metrics = metrics
        return metrics
    
    def _step(self, action: int) -> RLStepResult:
        sr = super()._step(action)
        # current_pose = self.controller.last_event.metadata["agent"]
        # heading_idx = int(np.round(current_pose["rotation"]["y"] / 30)) % 11
        # possible_headings = [0.0]*12 
        # possible_headings[heading_idx] = 1
        # map_update = [current_pose['position']['x'],current_pose['position']['z'],1]
        # self.aggregate_map = update_aggregate_map_blocky(self.aggregate_map,[map_update],**self.map_scaling_params)
        # # self.instant_map = map_utils.populate_instantaneous_map(self.instant_map,[map_update+possible_headings])

        # ## New objects
        # visible_object_ids = set(
        #     o["objectId"]
        #     for o in self.controller.last_event.metadata["objects"]
        #     if o["visible"]
        #     and o["objectType"] not in unrewarded_objects
        # )

        
        # self.seen_object_ids.update(visible_object_ids)
        
        # # consecutive move aheads
        # if action == 1:
        #     self.consecutive_move_aheads[-1] += 1
        # else:
        #     self.consecutive_move_aheads.append(0)
            
        # # visited pose
        # if self.controller.last_event.metadata["agent"]["position"] in self.path[:-1]:
        #     self.num_visited_pos += 1
        
        
        # ## New rooms
        # if self.visited_rooms is None:
        #     self.visited_rooms, _, self.current_room_id = rooms_visited(self.path,self.room_polys)
        # else:
        #     self.visited_rooms,previously_visited_rooms, self.current_room_id = rooms_visited(self.path[-3:],self.room_polys,self.visited_rooms)
        # if not self.ordered_visited_room_ids or self.ordered_visited_room_ids[-1] != self.current_room_id:
        #     self.ordered_visited_room_ids.append(self.current_room_id)
            

        # ## Map video
        # if self.visualize:
        #     self.map_observations.append(self.sensor_suite.sensors['aggregate_map_sensor'].get_observation(self.controller,self))

        return sr

class ProcTHORFleeNavTaskSamplerEval(ProcTHORFleeNavTaskSampler):
    def __init__(
        self,
        args: TaskSamplerArgs,
        flee_nav_task_type: Type = ProcTHORFleeNavTask,
        **extra_task_kwargs,
    ) -> None:
        self.args = args
        self.extra_task_kwargs = extra_task_kwargs
        self.flee_nav_task_type = flee_nav_task_type
        random.shuffle(self.args.house_inds)

        self.controller: Optional[Controller] = None
        self.distance_cache = DynamicDistanceCache(rounding=1)

        # get the total number of tasks assigned to this process
        self.reset_tasks = self.args.max_tasks

        self.resample_same_scene_freq = args.resample_same_scene_freq
        """The number of times to resample the same houses before moving to the next on."""

        self.house_inds_index = 0
        self.episode_index = 0

        self._last_sampled_task: Optional[ProcTHORFleeNavTask] = None

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

        self.reset()
        self.counter = 0

    def set_seed(self, seed: int):
        set_seed(seed)
        

    def next_task(
        self, force_advance_scene: bool = False
    ) -> Optional[ProcTHORFleeNavTask]:
        # NOTE: Stopping condition
        if self.args.max_tasks <= 0:
            return None

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

        self.controller.step(action="ResetMaterials", raise_for_failure=True)

        door_ids = [door["id"] for door in self.house["doors"]]
        self.controller.step(
            action="SetObjectFilter",
            objectIds=door_ids, #target_object_ids + 
            raise_for_failure=True,
        )

        # NOTE: Set agent pose
        standing = (
            {}
            if self.args.controller_args["agentMode"] == "locobot"
            else {"standing": True}
        )
        # starting_pose = AgentPose(
        #     position=random.choice(self.reachable_positions),
        #     rotation=Vector3(x=0, y=random.choice(self.valid_rotations), z=0),
        #     horizon=30,
        #     **standing,
        # )
        starting_pose = AgentPose(
            position=self.reachable_positions[random_inds[self.counter] % len(self.reachable_positions)],
            rotation=Vector3(x=0, y=self.valid_rotations[random_inds[self.counter] % len(self.valid_rotations)], z=0),
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
                "starting_pose": starting_pose,
                "mirrored": self.args.allow_flipping and random.random() > 0.5,
            },
            # morl
            save_video=self.args.save_video,
            save_trajectory=self.args.save_trajectory,
            # objectives_combination=self.args.objectives_combination,
            objectives=self.args.objectives,
            valid_objectives=self.args.valid_objectives,
            adaptive_reward=self.args.adaptive_reward,
            reward_weights=self.args.reward_weights,
        )
        task_kwargs.update(self.extra_task_kwargs)
        self._last_sampled_task = self.flee_nav_task_type(**task_kwargs)
        self.counter = (self.counter + 1) % 1000
        return self._last_sampled_task


class ProcTHORFleeNavTaskMappingSampler(ProcTHORFleeNavTaskSamplerEval):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.map_cache = dict()
        self.room_poly_cache = dict()
    
    def get_house_map(self):
        if (self.house_index in self.map_cache):
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]
        else:
            base_map, xyminmax, pixel_sizes = build_centered_map(self.house)
            self.map_cache[self.house_index] = {"map":base_map,"xyminmax":xyminmax, "pixel_sizes":pixel_sizes}
            self.room_poly_cache[self.house_index],_ = get_rooms_polymap_and_type(self.house)
            return self.map_cache[self.house_index],self.room_poly_cache[self.house_index]

    def next_task(self, force_advance_scene: bool = False) -> Optional[ProcTHORFleeNavMappingTask]:
        parent_task = super().next_task(force_advance_scene)
        # if parent_task is not None:
        #     parent_task.set_map(*self.get_house_map())
        self._last_sampled_task = parent_task
        return self._last_sampled_task