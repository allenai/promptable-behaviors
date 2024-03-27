import os
try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
from omegaconf import DictConfig
from collections import defaultdict
from typing import Any, Dict, List, Literal
import math
import torch
from PIL import Image
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import wandb
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


class WandbLogging(Callback):
    def __init__(self):
        # NOTE: Makes it more statistically meaningful
        self.aggregate_by_means_across_n_runs: int = 10
        self.by_means_iter: int = 0
        self.by_metrics = dict()
        self.cfg = None
        self.latest_log_step = 0
        self.logging_path = ""



    def setup(self, name: str, **kwargs) -> None:
        self.cfg = kwargs["config"].cfg
        self.logging_path = f'{self.cfg.output_dir}/logs/{self.cfg.wandb.name}'

        wandb.init(
            project=kwargs["config"].cfg.wandb.project,
            entity=kwargs["config"].cfg.wandb.entity,
            name=name if kwargs["config"].cfg.wandb.name is None else kwargs["config"].cfg.wandb.name,
            config=self.cfg, #kwargs,
            dir=kwargs["config"].cfg.wandb.dir,
        )

    @staticmethod
    def get_columns(task: Dict[str, Any]) -> List[str]:
        """Get the columns of the quantitative table."""
        types = int, float, str, bool, wandb.Image, wandb.Video

        columns = []
        for key in task.keys():
            if isinstance(task[key], types):
                columns.append(key)

        for key in task["task_info"]:
            if isinstance(task["task_info"][key], types):
                columns.append(f"task_info/{key}")
        return columns

    @staticmethod
    def get_quantitative_table(tasks_data: List[Any], step: int) -> wandb.Table:
        """Get the quantitative table."""

        if len(tasks_data) == 0:
            return wandb.Table()

        data = []
        columns = WandbLogging.get_columns(tasks_data[0])
        columns.insert(0, "step")
        columns.insert(0, "path")
        columns.insert(0, "observations")

        for task in tasks_data:
            frames = task["observations"]
            frames_with_progress = []

            # NOTE: add progress bars
            for i, frame in enumerate(frames):
                # NOTE: flip the images if the task is mirrored
                if "mirrored" in task["task_info"] and task["task_info"]["mirrored"]:
                    frame = np.fliplr(frame)
                BORDER_SIZE = 15

                frame_with_progress = np.full(
                    (
                        frame.shape[0] + 50 + BORDER_SIZE * 2,
                        frame.shape[1] + BORDER_SIZE * 2,
                        frame.shape[2],
                    ),
                    fill_value=255,
                    dtype=np.uint8,
                )

                # NOTE: add border for action failures
                if i > 1 and not task["task_info"]["action_successes"][i - 1]:
                    frame_with_progress[0 : BORDER_SIZE * 2 + frame.shape[0]] = (
                        255,
                        0,
                        0,
                    )

                # NOTE: add the agent image
                frame_with_progress[
                    BORDER_SIZE : BORDER_SIZE + frame.shape[0],
                    BORDER_SIZE : BORDER_SIZE + frame.shape[1],
                ] = frame

                # NOTE: add the progress bar
                progress_bar = frame_with_progress[-35:-15, BORDER_SIZE:-BORDER_SIZE]
                progress_bar[:] = (225, 225, 225)
                if len(frames) > 1:
                    num_progress_pixels = int(
                        progress_bar.shape[1] * i / (len(frames) - 1)
                    )
                    progress_bar[:, :num_progress_pixels] = (38, 94, 212)

                frames_with_progress.append(frame_with_progress)

            frames = np.stack(frames_with_progress, axis=0)
            frames = np.moveaxis(frames, [1, 2, 3], [2, 3, 1])
            trajectory = wandb.Video(frames, fps=5, format="mp4")

            entry = []
            for column in columns:
                if column == "observations":
                    entry.append(trajectory)
                elif column == "step":
                    entry.append(step)
                elif column == "path":
                    entry.append(wandb.Image(task["path"]))
                elif column.startswith("task_info/"):
                    entry.append(task["task_info"][column[len("task_info/") :]])
                else:
                    entry.append(task[column])

            data.append(entry)

        # clean up column names
        columns = [
            c[len("task_info/") :] if c.startswith("task_info/") else c for c in columns
        ]

        return wandb.Table(data=data, columns=columns)

    def on_train_log(
        self,
        metrics: List[Dict[str, Any]],
        metric_means: Dict[str, float],
        step: int,
        tasks_data: List[Any],
        **kwargs,
    ) -> None:
        """Log the train metrics to wandb."""
        # quantitative_table = {}
        # if len(tasks_data) > 0 and "task_info" in tasks_data[0].keys():
        #     table = self.get_quantitative_table(tasks_data=tasks_data, step=step)
        #     quantitative_table = (
        #         {f"train-quantitative-examples/{step:012}": table} if table.data else {}
        #     )

        # for episode in metrics:
        #     by_rooms_key = (
        #         f"train-metrics-by-rooms/{episode['task_info']['rooms']}-rooms"
        #     )
        #     by_obj_type_key = (
        #         f"train-metrics-by-obj-type/{episode['task_info']['object_type']}"
        #     )

        #     for k in (by_rooms_key, by_obj_type_key):
        #         if k not in self.by_metrics:
        #             self.by_metrics[k] = {
        #                 "means": {
        #                     "reward": 0,
        #                     "ep_length": 0,
        #                     "success": 0,
        #                     "spl": 0,
        #                     "dist_to_target": 0,
        #                 },
        #                 "count": 0,
        #             }
        #         self.by_metrics[k]["count"] += 1
        #         for metric in self.by_metrics[k]["means"]:
        #             old_mean = self.by_metrics[k]["means"][metric]
        #             self.by_metrics[k]["means"][metric] = (
        #                 old_mean
        #                 + (episode[metric] - old_mean) / self.by_metrics[k]["count"]
        #             )

        by_means_dict = {}
        self.by_means_iter += 1
        if self.by_means_iter % self.aggregate_by_means_across_n_runs == 0:
            # NOTE: log by means
            for metric, info in self.by_metrics.items():
                for mean_key, mean in info["means"].items():
                    key = f"/{mean_key}-".join(metric.split("/"))
                    by_means_dict[key] = mean
            # NOTE: reset the means
            self.by_metrics = dict()

        if self.cfg.logging.plot_codebook and (
                step > self.latest_log_step + self.cfg.logging.frequency * self.cfg.machine.num_train_processes):
            self.latest_log_step = step
            codebook_probs = np.load(f'{self.logging_path}/codebook_probs_latest.npy')
            codebook = torch.load(f'{self.logging_path}/codebook_latest.pt').mean(dim=1).detach().cpu().numpy()
            codebook_probs_image = plot_hmap(1 - codebook_probs, log_path=self.logging_path, plot_name="codebook probs")
            codebook_image = plot_hmap(codebook, log_path=self.logging_path, plot_name="codebook")

            wandb.log(
                {
                    **metric_means,
                    **by_means_dict,
                    **quantitative_table,
                    "step": step,
                    "codebook probs": codebook_probs_image,
                    "codebook": codebook_image,
                },
                step=step
            )
        else:

            wandb.log(
                {
                    **metric_means,
                    **by_means_dict,
                    # **quantitative_table,
                    "step": step,
                },
                step=step
            )


    @staticmethod
    def get_metrics_table(tasks: List[Any]) -> wandb.Table:
        """Get the metrics table."""
        columns = WandbLogging.get_columns(tasks[0])
        data = []
        for task in tasks:
            entry = []
            for column in columns:
                if column.startswith("task_info/"):
                    entry.append(task["task_info"][column[len("task_info/") :]])
                else:
                    if column in task:
                        entry.append(task[column])
                    else:
                        # valid_columns.remove(column)
                        # skip this entry
                        continue
            data.append(entry)

        columns = [
            c[len("task_info/") :] if c.startswith("task_info/") else c for c in columns
        ]
        # valid columns if not all columns are present in data
        # valid_columns = [column for column in columns if column in data[0]]
        return wandb.Table(data=data, columns=columns)

    @staticmethod
    def get_metric_plots(
        metrics: Dict[str, Any], split: Literal["valid", "test"], step: int
    ) -> Dict[str, Any]:
        """Get the metric plots."""
        plots = {}
        table = WandbLogging.get_metrics_table(metrics["tasks"])

        # NOTE: Log difficulty SPL and success rate
        if "difficulty" in metrics["tasks"][0]["task_info"]:
            plots[f"{split}-success-by-difficulty-{step:012}"] = wandb.plot.bar(
                table,
                "difficulty",
                "success",
                title=f"{split} Success by Difficulty ({step:,} steps)",
            )
            plots[f"{split}-spl-by-difficulty-{step:012}"] = wandb.plot.bar(
                table,
                "difficulty",
                "spl",
                title=f"{split} SPL by Difficulty ({step:,} steps)",
            )

        # NOTE: Log object type SPL and success rate
        if "object_type" in metrics["tasks"][0]["task_info"]:
            plots[f"{split}-success-by-object-type-{step:012}"] = wandb.plot.bar(
                table,
                "object_type",
                "success",
                title=f"{split} Success by Object Type ({step:,} steps)",
            )
            plots[f"{split}-spl-by-object-type-{step:012}"] = wandb.plot.bar(
                table,
                "object_type",
                "spl",
                title=f"{split} SPL by Object Type ({step:,} steps)",
            )

        return plots

    @staticmethod
    def get_by_scene_dataset_log(
        metrics: Dict[str, Any], split: Literal["train", "val", "test"]
    ) -> Dict[str, float]:
        by_scene_data = defaultdict(
            lambda: {
                "count": 0,
                "means": {
                    "reward": 0,
                    "ep_length": 0,
                    "success": 0,
                    "spl": 0,
                    "dist_to_target": 0,
                },
            }
        )
        if (
            len(metrics["tasks"]) > 0
            and "sceneDataset" in metrics["tasks"][0]["task_info"]
        ):
            for task in metrics["tasks"]:
                scene_dataset = task["task_info"]["sceneDataset"]
                by_scene_data[scene_dataset]["count"] += 1
                for key in by_scene_data[scene_dataset]["means"]:
                    old_mean = by_scene_data[scene_dataset]["means"][key]
                    value = float(task[key])
                    by_scene_data[scene_dataset]["means"][key] = (
                        old_mean
                        + (value - old_mean) / by_scene_data[scene_dataset]["count"]
                    )
        by_scene_data_log = {}
        for scene_dataset in by_scene_data:
            for mean, value in by_scene_data[scene_dataset]["means"].items():
                by_scene_data_log[f"{split}-{scene_dataset}/{mean}"] = value

        return by_scene_data_log

    def on_valid_log(
        self,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        checkpoint_file_name: str,
        tasks_data: List[Any],
        step: int,
        **kwargs,
    ) -> None:
        """Log the validation metrics to wandb."""

        # by_scene_dataset_log = WandbLogging.get_by_scene_dataset_log(
        #     metrics, split="val"
        # )
        # plots = (
        #     self.get_metric_plots(metrics=metrics, split="valid", step=step)
        #     if metrics
        #     else {}
        # )
        # val_table = {}
        # if len(tasks_data) > 0 and "task_info" in tasks_data[0].keys():
        #     table = self.get_quantitative_table(tasks_data=tasks_data, step=step)
        #     val_table = (
        #         {f"valid-quantitative-examples/{step:012}": table} if table.data else {}
        #     )
        wandb.save(checkpoint_file_name)
        wandb.log(
            {**metric_means, "step": step,}, #**plots, **by_scene_dataset_log, "step": step, **val_table},
            step=step
        )

    def on_test_log(
        self,
        checkpoint_file_name: str,
        metrics: Dict[str, Any],
        metric_means: Dict[str, float],
        tasks_data: List[Any],
        step: int,
        **kwargs,
    ) -> None:
        """Log the test metrics to wandb."""

        # by_scene_dataset_log = WandbLogging.get_by_scene_dataset_log(
        #     metrics, split="test"
        # )
        # plots = (
        #     self.get_metric_plots(metrics=metrics, split="test", step=step)
        #     if metrics
        #     else {}
        # )

        # test_table = {}
        # if len(tasks_data) > 0 and "task_info" in tasks_data[0].keys():
        #     table = self.get_quantitative_table(tasks_data=tasks_data, step=step)
        #     test_table = (
        #         {f"test-quantitative-examples/{step:012}": table} if table.data else {}
        #     )

        wandb.log(
            {
                **metric_means,
                # **plots,
                # **by_scene_dataset_log,
                "step": step,
                # **test_table
            },
            step=step
        )
