import os
import cv2
import imageio
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)

def save_video(frames, filename, fps=10):
    # video_name = os.path.join(video_dir, "{}_{}.mp4".format(env_name, version))
    video_name = filename
    with imageio.get_writer(video_name, fps=fps) as writer:
        im_shape = frames[-1].shape
        for im in frames:
            # convert BGR to RGB
            # im = im[:, :, ::-1]
            if (im.shape[0] != im_shape[0]) or (im.shape[1] != im_shape[1]):
                im = cv2.resize(im, (im_shape[1], im_shape[0]))
            writer.append_data(im.astype(np.uint8))
        writer.close()
    print("Video saved to {}".format(video_name))

def add_label(frame, label, font_scale=1, thickness=2):
    """Add label to the bottom center of a frame."""
    height, width, _ = frame.shape
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = height - 10
    cv2.putText(frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 255), thickness)

def add_label_below(frame, label, height=50, font_scale=1, thickness=2, background_color=(0, 0, 0)):
    """Add label below a frame."""
    text_width, text_height = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
    label_frame = np.full((height, frame.shape[1], 3), background_color, dtype=np.uint8)
    text_x = (frame.shape[1] - text_width) // 2
    text_y = (height + text_height) // 2
    cv2.putText(label_frame, label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    return np.vstack([frame, label_frame])

def add_separator(height, width=10):
    """Return a black separator of specified height and width."""
    return np.zeros((height, width, 3), dtype=np.uint8)

def concatenate_videos_with_labels_side_by_side(video_paths, labels, output_path, separator_width=10, episode=None):
    video_caps = [cv2.VideoCapture(vp) for vp in video_paths]
    video_widths = [int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)) for vc in video_caps]
    video_heights = [int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)) for vc in video_caps]
    video_fps = [int(vc.get(cv2.CAP_PROP_FPS)) for vc in video_caps]
    max_duration_frames = max(int(vc.get(cv2.CAP_PROP_FRAME_COUNT)) for vc in video_caps)

    target_width = min(video_widths)
    target_height = min(video_heights)
    separator_width = 10

    # Account for the added width due to separators
    final_width = target_width * len(video_caps) + separator_width * (len(video_caps) - 1)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, video_fps[0], (final_width, target_height + 50))  # added +50 for label height

    individual_frame_indices = [0 for _ in video_caps]

    for _ in range(max_duration_frames):
        frames = []
        for idx, vc in enumerate(video_caps):
            ret, frame = vc.read()
            if not ret:
                vc.set(cv2.CAP_PROP_POS_FRAMES, vc.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
                _, frame = vc.read()
            else:
                individual_frame_indices[idx] += 1

            if frame is None:
                # use last frame
                frame = frames[-1]
            
            frame_resized = cv2.resize(frame, (target_width, target_height))
            add_label(frame_resized, str(individual_frame_indices[idx]))
            frame_with_label = add_label_below(frame_resized, labels[idx])
            frames.append(frame_with_label)

            if idx < len(video_caps) - 1:  # Not for the last video
                frames.append(add_separator(frame_with_label.shape[0], separator_width))

        final_frame = np.hstack(frames)
        if episode is not None:
            final_frame = add_label_below(final_frame, episode)
        out.write(final_frame)

    # Cleanup
    for vc in video_caps:
        vc.release()
    out.release()
