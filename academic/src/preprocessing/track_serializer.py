import os
import json
from collections import defaultdict
from tqdm import tqdm
import numpy as np


class TrackSerializer:

    def __init__(self, tracks_dir):

        self.tracks_dir = tracks_dir

    def serialize_video_tracks(self, video_folder):

        track_file = os.path.join(self.tracks_dir, video_folder, "tracks.txt")

        if not os.path.exists(track_file):
            print(f"Skipping {video_folder}, tracks.txt not found")
            return

        tracks = defaultdict(list)

        with open(track_file, "r") as f:

            lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) < 6:
                    continue

                frame_id = int(parts[0])
                track_id = int(parts[1])

                x1 = float(parts[2])
                y1 = float(parts[3])
                x2 = float(parts[4])
                y2 = float(parts[5])

                tracks[track_id].append({
                    "frame": frame_id,
                    "bbox": [x1, y1, x2, y2]
                })

        output_file = os.path.join(self.tracks_dir, video_folder, "tracks.json")

        with open(output_file, "w") as f:
            json.dump(tracks, f, indent=4)

        print(f"Serialized tracks for {video_folder}")

    def run(self):

        video_folders = os.listdir(self.tracks_dir)

        print(f"Found {len(video_folders)} video folders")

        for video in tqdm(video_folders):

            video_path = os.path.join(self.tracks_dir, video)

            if not os.path.isdir(video_path):
                continue

            self.serialize_video_tracks(video)


def build_detection_features(video_name, tracks_dir="data/processed/tracks", clip_size=16):
    """
    Convert tracks.json → detections_per_clip

    clip_size = number of frames per clip (must match your clip_generator)
    """

    track_file = os.path.join(tracks_dir, video_name, "tracks.json")

    if not os.path.exists(track_file):
        print(f"No tracks.json found for {video_name}")
        return []

    with open(track_file, "r") as f:
        tracks = json.load(f)

    # frame → list of track_ids
    frame_map = defaultdict(list)

    for track_id, track_data in tracks.items():

        for entry in track_data:

            frame_id = entry["frame"]
            frame_map[frame_id].append(int(track_id))

    if not frame_map:
        return []
    sorted_frames = sorted(frame_map.keys())
    detections_per_clip = []
    for i in range(0, len(sorted_frames), clip_size):
        clip_frames = sorted_frames[i:i+clip_size]
        frame_counts = []
        for f in clip_frames:
            frame_counts.append(len(frame_map[f]))
        if len(frame_counts) == 0:
            num_people = 0.0
        else:
            avg = np.mean(frame_counts)

            # normalize relative to clip
            max_count = max(frame_counts) + 1e-5
            num_people = avg / max_count   # value between 0–1

            # scale to useful range
            num_people = num_people * 3

        num_people = min(num_people, 6)
        detections_per_clip.append({
            "num_people": num_people,
            "avg_conf": 0.8
        })
    return detections_per_clip

if __name__ == "__main__":

    tracks_dir = "data/processed/tracks"

    serializer = TrackSerializer(tracks_dir)

    serializer.run()