from typing import Any, Dict, List, Sequence, Optional, Set
import numpy as np
import gym
from PIL import Image
from allenact.base_abstractions.sensor import Sensor
from utils.local_logging import unnormalize_image, WandbLoggingSensor

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import torch
from PIL import Image
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import math
import wandb
import os
from allenact.base_abstractions.callbacks import Callback

def plot_hmap(m, log_path, plot_name):
    size = int(math.sqrt(m.shape[0]))
    fig = plt.figure()
    sns.set(rc={'figure.figsize':(8,6)})
    ax = sns.heatmap(m[:size*size].reshape(size, size), linewidth=0.5)
    plt.savefig(f'{log_path}/{plot_name}.png', dpi=400)
    image = wandb.Image(
        np.array(Image.open(f'{log_path}/{plot_name}.png')),
        caption=plot_name
    )
    return image

class SimpleWandbLogging(Callback):
    def __init__(
        self,
        project: str,
        entity: str,
        plot_codebook=False,
        logging_path="",
        logging_frequency=1e3,
    ):
        self.project = project
        self.entity = entity

        self._defined_metrics: Set[str] = set()
        self.plot_codebook = plot_codebook
        self.logging_frequency = logging_frequency
        self.latest_log_step = 0
        self.logging_path = logging_path


    def setup(self, name: str, **kwargs) -> None:
        wandb.init(
            project=self.project,
            entity=self.entity,
            name=name,
            config=kwargs,
        )

    def _define_missing_metrics(
        self,
        metric_means: Dict[str, float],
        scalar_name_to_total_experiences_key: Dict[str, str],
    ):
        for k, v in metric_means.items():
            if k not in self._defined_metrics:
                wandb.define_metric(
                    k,
                    step_metric=scalar_name_to_total_experiences_key.get(k, "training_step"),
                )

                self._defined_metrics.add(k)

    def on_train_log(
        self,
        metrics: List[Dict[str, Any]],
        metric_means: Dict[str, float],
        step: int,
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        **kwargs,
    ) -> None:
        """Log the train metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        if self.plot_codebook and (
                step > self.latest_log_step + self.logging_frequency):
            self.latest_log_step = step
            codebook_probs = np.load(f'{self.logging_path}/codebook_probs_latest.npy')
            codebook = torch.load(f'{self.logging_path}/codebook_latest.pt').mean(dim=1).detach().cpu().numpy()
            codebook_probs_image = plot_hmap(1 - codebook_probs, log_path=self.logging_path, plot_name="codebook probs")
            codebook_image = plot_hmap(codebook, log_path=self.logging_path, plot_name="codebook")

            wandb.log(
                {
                    **metric_means,
                    "training_step": step,
                    "codebook probs": codebook_probs_image,
                    "codebook": codebook_image,
                },
                step=step
            )
        else:

            wandb.log(
                {
                    **metric_means,
                    "training_step": step,
                },
                step=step
            )

    def combine_rgb_across_episode(self, observation_list):
        all_rgb = []
        for obs in observation_list:
            frame = unnormalize_image(obs["rgb"])
            all_rgb.append(np.array(Image.fromarray((frame * 255).astype(np.uint8))))

        return all_rgb

    def on_valid_log(
        self,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        checkpoint_file_name: str,
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        step: int,
        **kwargs,
    ) -> None:
        """Log the validation metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        wandb.log(
            {
                **metric_means,
                "training_step": step,
            },
            step=step
        )

    def get_table_content(self, metrics, tasks_data, frames_with_logit_flag=False):
        observation_list = [
            tasks_data[i]["local_logging_callback_sensor"]["observations"]
            for i in range(len(tasks_data))
        ]  # NOTE: List of episode frames

        list_of_video_frames = [self.combine_rgb_across_episode(obs) for obs in observation_list]

        path_list = [
            tasks_data[i]["local_logging_callback_sensor"]["path"] for i in range(len(tasks_data))
        ]  # NOTE: List of path frames

        if frames_with_logit_flag:
            frames_with_logits_list_numpy = [
                tasks_data[i]["local_logging_callback_sensor"]["frames_with_logits"]
                for i in range(len(tasks_data))
            ]

        table_content = []
        frames_with_logits_list = []

        videos_without_logits_list = []

        for idx, data in enumerate(zip(list_of_video_frames, path_list, metrics["tasks"])):
            frames_without_logits, path, metric_data = (
                data[0],
                data[1],
                data[2],
            )

            wandb_data = (
                wandb.Video(
                    np.moveaxis(np.array(frames_without_logits), [0, 3, 1, 2], [0, 1, 2, 3]),
                    fps=10,
                    format="mp4",
                ),
                wandb.Image(path[0]),
                metric_data["ep_length"],
                metric_data["success"],
                metric_data["dist_to_target"],
                metric_data["task_info"]["task_type"],
                metric_data["task_info"]["house_name"],
                metric_data["task_info"]["target_object_type"],
                metric_data["task_info"]["id"],
                idx,
            )

            videos_without_logits_list.append(
                wandb.Video(
                    np.moveaxis(np.array(frames_without_logits), [0, 3, 1, 2], [0, 1, 2, 3]),
                    fps=5,
                    format="mp4",
                ),
            )

            table_content.append(wandb_data)
            if frames_with_logit_flag:
                frames_with_logits_list.append(
                    wandb.Video(np.array(frames_with_logits_list_numpy[idx]), fps=10, format="mp4")
                )

        return table_content, frames_with_logits_list, videos_without_logits_list

    def on_test_log(
        self,
        checkpoint_file_name: str,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        scalar_name_to_total_experiences_key: Dict[str, str],
        step: int,
        **kwargs,
    ) -> None:
        """Log the test metrics to wandb."""

        self._define_missing_metrics(
            metric_means=metric_means,
            scalar_name_to_total_experiences_key=scalar_name_to_total_experiences_key,
        )

        if tasks_data[0]["local_logging_callback_sensor"] is not None:
            frames_with_logits_flag = False

            if "frames_with_logits" in tasks_data[0]["local_logging_callback_sensor"]:
                frames_with_logits_flag = True

            (
                table_content,
                frames_with_logits_list,
                videos_without_logit_list,
            ) = self.get_table_content(metrics, tasks_data, frames_with_logits_flag)

            video_dict = {"all_videos": {}}

            for vid, data in zip(frames_with_logits_list, table_content):
                idx = data[-1]
                video_dict["all_videos"][idx] = vid

            table = wandb.Table(
                columns=[
                    "Trajectory",
                    "Path",
                    "Episode Length",
                    "Success",
                    "Dist to target",
                    "Task Type",
                    "House Name",
                    "Target Object Type",
                    "Task Id",
                    "Index",
                ]
            )

            for data in table_content:
                table.add_data(*data)

            # TODO: Add with logit videos separately

            wandb.log(
                {
                    **metric_means,
                    "training_step": step,
                    "Qualitative Examples": table,
                    "Videos": video_dict,
                },
                step=step
            )
        else:
            wandb.log(
                {
                    **metric_means,
                    "training_step": step,
                    # "Qualitative Examples": table,
                    # "Videos": video_dict,
                },
                step=step
            )

    def after_save_project_state(self, base_dir: str) -> None:
        pass

    def callback_sensors(self) -> Optional[Sequence[Sensor]]:
        return [
            WandbLoggingSensor(
                uuid="local_logging_callback_sensor", observation_space=gym.spaces.Discrete(1)
            ),
        ]

        return [
            LocalLoggingSensor(
                uuid="local_logging_callback_sensor", observation_space=gym.spaces.Discrete(1)
            ),
        ]
