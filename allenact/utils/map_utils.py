import numpy as np
import cv2
from allenact.utils.video_utils import add_label_below

def reachable_positions_to_matrix(reachable_positions, grid_size=0.25):
    min_x, max_x, min_y, max_y = reachable_positions[:, 0].min(), reachable_positions[:, 0].max(), reachable_positions[:, 1].min(), reachable_positions[:, 1].max()
    width = int((max_x - min_x) / grid_size)
    height = int((max_y - min_y) / grid_size)
    reachable_positions_matrix = np.ones((width+1, height+1)) # add 1 because each grid corresponds to a point
    converted_positions = np.array([(pos[0] - min_x, pos[1] - min_y) for pos in reachable_positions])
    converted_positions = (converted_positions / grid_size).astype(int)
    reachable_positions_matrix[converted_positions[:, 0], converted_positions[:, 1]] = 0
    return reachable_positions_matrix

def position_to_grid(position, min_x, min_y, grid_size):
    return int((position['x'] - min_x) / grid_size), int((position['z'] - min_y) / grid_size)

def grid_to_position(grid, min_x, min_y, grid_size):
    return grid[0] * grid_size + min_x, grid[1] * grid_size + min_y

def draw_agent(reachable_positions_matrix, agent_position, min_x, min_y, safety_grid=0, draw_safety_grid=True, save_image=False, label=None, return_safety=False):
    # use cv2 to draw a red dot on the agent's position
    agent_grid = position_to_grid(agent_position, min_x=min_x, min_y=min_y, grid_size=0.25)
    rgb_reachable_positions_matrix = cv2.cvtColor(np.array(reachable_positions_matrix*255, dtype='uint8'), cv2.COLOR_GRAY2RGB)

    # draw red square around agent
    # count ones in safety grid
    min_x_safety_grid = max(0, agent_grid[0]-safety_grid)
    max_x_safety_grid = min(rgb_reachable_positions_matrix.shape[0], agent_grid[0]+safety_grid+1)
    min_y_safety_grid = max(0, agent_grid[1]-safety_grid)
    max_y_safety_grid = min(rgb_reachable_positions_matrix.shape[1], agent_grid[1]+safety_grid+1)
    
    count = np.sum(reachable_positions_matrix[min_x_safety_grid:max_x_safety_grid, min_y_safety_grid:max_y_safety_grid])
    safety_reward = count * 1
    
    if draw_safety_grid:
        rgb_reachable_positions_matrix[min_x_safety_grid:max_x_safety_grid, min_y_safety_grid:max_y_safety_grid] = np.array([255, 0, 0])*0.5 + rgb_reachable_positions_matrix[min_x_safety_grid:max_x_safety_grid, min_y_safety_grid:max_y_safety_grid]*0.5
    rgb_reachable_positions_matrix[agent_grid[0], agent_grid[1]] = [0, 0, 255]

    # resize for visualization - scale up by 10
    scale = 20
    resized_rgb_reachable_positions_matrix = cv2.resize(rgb_reachable_positions_matrix, (rgb_reachable_positions_matrix.shape[1]*scale, rgb_reachable_positions_matrix.shape[0]*scale), interpolation=cv2.INTER_NEAREST)
    # add caption below the image
    resized_rgb_reachable_positions_matrix = add_label_below(resized_rgb_reachable_positions_matrix, "safety : {}".format(safety_reward), background_color=(100, 100, 100))
    if label is not None:
        resized_rgb_reachable_positions_matrix = add_label_below(resized_rgb_reachable_positions_matrix, label, background_color=(100, 100, 100))

    if save_image:
        if label is None:
            cv2.imwrite("../data/reachable_positions_agent_pos.png", resized_rgb_reachable_positions_matrix[:, :, ::-1])
        else:
            cv2.imwrite("../data/reachable_positions_agent_pos_{}.png".format(label), resized_rgb_reachable_positions_matrix[:, :, ::-1])

    if return_safety:
        return resized_rgb_reachable_positions_matrix, safety_reward
    return resized_rgb_reachable_positions_matrix