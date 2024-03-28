from shapely.geometry import Polygon
from shapely.ops import unary_union
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import json
from mpl_toolkits.axes_grid1 import Divider, Size
import PIL, io, copy

from .utils import ForkedPdb
matplotlib.use('agg') # non-interactive backend for thread-safety 

import cv2
from procthor_visual_nav.utils.video_utils import add_label_below

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

def generate_base_polygon(partial_house):
    all_room_points = [x['floorPolygon'] for x in partial_house['rooms']]
    all_room_polys = [Polygon([(p['x'], p['z']) for p in rp]) for rp in all_room_points]
    floor_poly = unary_union(all_room_polys)
    return all_room_polys, floor_poly

def get_interior_doors(partial_house):
    try:
        interior_doors = [x for x in partial_house['doors'] ]#if x.get('openness', 0) > 0]
    except:
        ForkedPdb().set_trace()
    # interior_doors = [x for x in partial_house['doors'] if x['openness'] > 0]
    for door in interior_doors:
        door['world_center'] = {}
        wall_poly = [x['polygon'] for x in partial_house['walls'] if x['id']==door['wall0']]
        door_dist = door['assetPosition']['x']
        try:
            p0,p1 = wall_poly[0][0],wall_poly[0][1]
        except:
            ForkedPdb().set_trace()
        # p0,p1 = wall_poly[0][0],wall_poly[0][1]
        stratio = door_dist / np.sqrt((p0['x'] - p1['x'])**2 + (p0['z'] - p1['z'])**2)
        door['world_center']['x'] = stratio * (p1['x'] - p0['x']) + p0['x']
        door['world_center']['z'] = stratio * (p1['z'] - p0['z']) + p0['z']
    return interior_doors

def build_centered_map(partial_house, pixel_scaling=6, mask_color=(1,0,0), wall_color=(1,1,0)):
    # 12 cm per pixel makes interior walls approx the right thickness
    plist,combined = generate_base_polygon(partial_house)
    doors = get_interior_doors(partial_house)
    xmin,xmax,ymin,ymax = np.min(combined.exterior.xy[0])-1,np.max(combined.exterior.xy[0])+1,np.min(combined.exterior.xy[1])-1,np.max(combined.exterior.xy[1])+1
    pixel_x,pixel_y = np.ceil(np.abs(xmax-xmin)*pixel_scaling), np.ceil(np.abs(ymax-ymin)*pixel_scaling)
    fig = plt.figure(figsize=(10, 10)) # 1024/standard DPI, changed later. use of fig and ax1 for thread-safety
    DPI = fig.get_dpi()
    h = [Size.Fixed(0.0), Size.Fixed(pixel_x/float(DPI)), Size.Fixed(0.0)]
    v = [Size.Fixed(0.0), Size.Fixed(pixel_y/float(DPI)), Size.Fixed(0.0)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax1 = fig.add_axes(divider.get_position(),
                  axes_locator=divider.new_locator(nx=1, ny=1))
    ax1.clear()
    # floor mask
    patch = patches.Polygon(np.array(combined.exterior.xy).T, fc=mask_color)
    ax1.add_patch(patch)
    # interior walls
    for p in plist:
        ax1.plot(*p.exterior.xy,color=wall_color)
    # interior doors as gaps (or dots for debugging)
    for d in doors:
        ax1.plot(d['world_center']['x'], d['world_center']['z'], color=mask_color, marker='.',linewidth=3)
    # square off and convert/verify size
    ax1.axis([xmin,xmax,ymin,ymax])
    ax1.axis('off')
    buffer_ = io.BytesIO()
    fig.savefig( buffer_, format = "png", pad_inches = 0, bbox_inches = 'tight')
    buffer_.seek(0)
    base_map = np.asarray(PIL.Image.open( buffer_ ))
    plt.close(fig)
    base_map[np.where((base_map==[255,255,255,255]).all(axis=2))] = [0,0,0,0] # white to black
    base_map[:,:,2] = 0
    base_map[:,:,3] = 0
    return np.round(base_map/255), [xmin,xmax,ymin,ymax], [pixel_x,pixel_y]

def centered_pixel_from_point(p, xyminmax, pixel_sizes):
    xmin,xmax,ymin,ymax = xyminmax
    row = int((ymax-p[1])/np.abs(ymax-ymin)*pixel_sizes[1]) # y becomes rows
    col = int((p[0]-xmin)/np.abs(xmax-xmin)*pixel_sizes[0]) # x becomes columns
    return (row,col)

def build_map(partial_house,pixel_size=1024, mask_color=(1,0,0), wall_color=(1,1,0)):
    plist,combined = generate_base_polygon(partial_house)
    doors = get_interior_doors(partial_house)
    fig = plt.figure(figsize=(10, 10)) # 1024/standard DPI, verified later. use of fig and ax1 for thread-safety
    DPI = fig.get_dpi()
    h = [Size.Fixed(0.0), Size.Fixed(pixel_size/float(DPI)), Size.Fixed(0.0)]
    v = [Size.Fixed(0.0), Size.Fixed(pixel_size/float(DPI)), Size.Fixed(0.0)]
    divider = Divider(fig, (0, 0, 1, 1), h, v, aspect=False)
    ax1 = fig.add_axes(divider.get_position(),
                  axes_locator=divider.new_locator(nx=1, ny=1))
    ax1.clear()
    # floor mask
    patch = patches.Polygon(np.array(combined.exterior.xy).T, fc=mask_color)
    ax1.add_patch(patch)
    # interior walls
    for p in plist:
        ax1.plot(*p.exterior.xy,color=wall_color)
    # interior doors as gaps (or dots for debugging)
    for d in doors:
        ax1.plot(d['world_center']['x'], d['world_center']['z'], color=mask_color, marker='.',linewidth=3)
    # square off and convert/verify size
    ax1.axis([-25, 25, -25, 25])
    ax1.axis('off')
    buffer_ = io.BytesIO()
    fig.savefig( buffer_, format = "png", pad_inches = 0, bbox_inches = 'tight')
    buffer_.seek(0)
    base_map = np.asarray(PIL.Image.open( buffer_ ))
    base_map[np.where((base_map==[255,255,255,255]).all(axis=2))] = [0,0,0,0] # white to black
    base_map[:,:,2] = 0
    base_map[:,:,3] = 0
    return base_map/255

# def pixel_from_point(p,pixel_size=1024):
#     return (int((25-p[0])/50*pixel_size),int((25+p[1])/50*pixel_size))

def update_aggregate_map(prev_map,pos_probs,xyminmax=[-25, 25, -25, 25], pixel_sizes=[1024,1024],max_num_steps=500):
    new_map = copy.deepcopy(prev_map) #TODO: necessary?
    # decay the probability. Set zero here to forget instead.
    new_map[:,:,3] = new_map[:,:,3] - 1/max_num_steps 
    # populate the current step probability component
    for prob in pos_probs:
        pixel_loc = centered_pixel_from_point(prob,xyminmax,pixel_sizes)
        new_map[pixel_loc[0],pixel_loc[1],3] = prob[2]
    # decay the temporal/argmax component
    new_map[:,:,2] = new_map[:,:,2] - 1/max_num_steps
    # this has an edge case of a tie where two values are returned - use the first.
    curr_guess_loc = np.where(new_map[:,:,3] == np.max(new_map[:,:,3])) 
    new_map[curr_guess_loc[0][0],curr_guess_loc[1][0],2] = 1
    return new_map.clip(0.0,1.0)

def update_aggregate_map_blocky(prev_map,pos_probs,xyminmax=[-25, 25, -25, 25], pixel_sizes=[1024,1024],max_num_steps=500):
    new_map = copy.deepcopy(prev_map)
    # Set zero here to forget.
    new_map[:,:,3] = 0
    # populate the current and aggregate step 
    for prob in pos_probs:
        pixel_loc = centered_pixel_from_point(prob,xyminmax,pixel_sizes)
        new_map[pixel_loc[0]-3:pixel_loc[0]+3,pixel_loc[1]-3:pixel_loc[1]+3,[2,3]] = prob[2]

    return new_map.clip(0.0,1.0)

# def populate_instantaneous_map(base_map,pos_probs):
#     # new_map = copy.deepcopy(base_map)
#     base_map[:,:,2:] = 0 # reset
#     for prob in pos_probs:
#         pixel_loc = pixel_from_point(prob)
#         base_map[pixel_loc[0],pixel_loc[1],3:] = prob[3:]
#     return base_map.clip(0.0,1.0)

# # test the functions and visualize
# scenes = 'local_scenes/all_back_apartment.json'
# with open(scenes) as f:
#     all_houses=json.load(f)

# h = all_houses[1]
# prev_map = build_map(h)
# p = [[0,0,1]] # gt dummy starting location
# for idx in range(500):
#     new_map = update_aggregate_map(prev_map,p)
#     p[0][0] += np.random.standard_normal()*0.05
#     p[0][1] += np.random.standard_normal()*0.05
#     prev_map = new_map

# h = all_houses[1]
# prev_map, xyminmax, pixel_sizes = build_centered_map(h)
# p = [[0,0,1]] # gt dummy starting location
# for idx in range(500):
#     new_map = update_aggregate_map(prev_map,p, xyminmax, pixel_sizes)
#     p[0][0] += np.random.standard_normal()*0.05
#     p[0][1] += np.random.standard_normal()*0.05
#     prev_map = new_map

# combo = np.sum(prev_map[:,:,[1,2]],axis=2)
# plt.clf()
# plt.imshow(combo)
# plt.savefig("combo.png")
