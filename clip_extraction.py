import cv2
import os
import glob
import numpy as np


def clip_extract(path, output="extracted", fps_target=6, clip_sec=5, stride_sec=2):
    os.makedirs(output, exist_ok=True)

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    # Resample to target FPS if needed
    if fps != fps_target:
        total_frames_original = len(frames)
        duration_sec = total_frames_original / fps
        target_frame_count = int(duration_sec * fps_target)
        indices = np.linspace(0, total_frames_original - 1, target_frame_count, dtype=int)
        frames = [frames[i] for i in indices]

    total_frames = len(frames)
    window_size = clip_sec * fps_target
    stride = stride_sec * fps_target
    start = 0
    clip_id = 0

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while start + window_size <= total_frames:
        clip_frames = frames[start:start + window_size]
        clip_path = os.path.join(output, f"{os.path.splitext(os.path.basename(path))[0]}_clip_{clip_id}.mp4")
        out = cv2.VideoWriter(clip_path, fourcc, fps_target, (width, height))
        for f in clip_frames:
            out.write(f)
        out.release()
        clip_id += 1
        start += stride

    # Handle tail frames if remaining >= 3 sec
    remaining = total_frames - start
    if remaining >= 3 * fps_target and total_frames >= window_size:
        clip_frames = frames[-window_size:]
        clip_path = os.path.join(output, f"{os.path.splitext(os.path.basename(path))[0]}_clip_{clip_id}.mp4")
        out = cv2.VideoWriter(clip_path, fourcc, fps_target, (width, height))
        for f in clip_frames:
            out.write(f)
        out.release()

    print(f"{os.path.basename(path)} -> Total 5-sec clips saved: {clip_id + 1}")



input_folder = "D:/Major project/dataset/SCVD/Test/Normal"
output_folder = "D:/Major project/dataset/converted"

video_files = glob.glob(os.path.join(input_folder, "*.*"))  # Supports all video formats

for video_file in video_files:
    clip_extract(video_file, output_folder)
