import os
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import six

def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                    header_color='#40466e', row_colors=['#f1f1f2', 'w'], edge_color='w',
                    bbox=[0, 0, 1, 1], header_columns=0, diff_color_columns=[],diff_color='g',red_cell_idx=[],
                    ax=None, **kwargs):
    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')

    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, **kwargs)

    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)
    # center align
    
    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight='bold', color='w')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

        if k[0] == 0 and k[1] in diff_color_columns:
            cell.set_facecolor(diff_color)
            cell.set_text_props(weight='bold', color='w')

        if [k[0], k[1]] in red_cell_idx:
            # red text
            cell.set_text_props(weight='bold', color='r')
    return ax

def get_eval_results(videos_dir, extension="mp4", episode_list=None):
    videos_path_list = glob(os.path.join(videos_dir, "**/*.{}".format(extension)), recursive=True)
    eval_results = {}

    for video_path in videos_path_list:
        video_path = video_path.split("/")[-1]
        episode = video_path.split("/")[-1].split("_weights")[0]
        if episode_list is not None and episode not in episode_list:
            continue
        success = video_path.split("sr")[-1].split("_")[0]
        spl = video_path.split("spl")[-1].split("_")[0]
        dist_to_goal = video_path.split("dist")[-1].split("_")[0]
        ep_len = video_path.split("eplen")[-1].split("_")[0]
        reward = video_path.split("rewards")[-1].split("_")[0]
        weights = video_path.split("weights")[-1].split("_sr")[0] #.split("_")
        sub_rewards = video_path.split("subrewards")[-1].split(".{}".format(extension))[0].split("_")
        sub_rewards = [float(sub_reward) for sub_reward in sub_rewards]
        if episode not in eval_results:
            eval_results[episode] = {weights: {"success": success, "spl": spl, "dist_to_target": dist_to_goal, "episode_length": ep_len,\
                                               "reward": reward, "sub_rewards": sub_rewards}}
        else:
            eval_results[episode][weights] = {"success": success, "spl": spl, "dist_to_target": dist_to_goal, "episode_length": ep_len,\
                                               "reward": reward, "sub_rewards": sub_rewards}


    return eval_results