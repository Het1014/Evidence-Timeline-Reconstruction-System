import os
import cv2
import json
from tqdm import tqdm


class ClipGenerator:

    def __init__(self, frames_dir, tracks_dir, output_dir,
                 clip_length=8, stride=4):

        self.frames_dir = frames_dir
        self.tracks_dir = tracks_dir
        self.output_dir = output_dir

        self.clip_length = clip_length
        self.stride = stride

        os.makedirs(self.output_dir, exist_ok=True)

    def generate_clips_for_video(self, video_name):

        frame_folder = os.path.join(self.frames_dir, video_name)
        track_file = os.path.join(self.tracks_dir, video_name, "tracks.json")

        if not os.path.exists(track_file):
            print(f"Skipping {video_name} (tracks.json not found)")
            return

        output_video_folder = os.path.join(self.output_dir, video_name)
        os.makedirs(output_video_folder, exist_ok=True)

        with open(track_file, "r") as f:
            tracks = json.load(f)

        frame_files = sorted(os.listdir(frame_folder))
        total_frames = len(frame_files)

        clip_id = 0

        for start in range(0, total_frames - self.clip_length + 1, self.stride):

            end = start + self.clip_length

            clip_frames = frame_files[start:end]

            first_frame_path = os.path.join(frame_folder, clip_frames[0])
            frame = cv2.imread(first_frame_path)

            height, width, _ = frame.shape

            clip_path = os.path.join(
                output_video_folder,
                f"clip_{clip_id:04d}.mp4"
            )

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            out = cv2.VideoWriter(
                clip_path,
                fourcc,
                5,
                (width, height)
            )

            for frame_name in clip_frames:

                frame_path = os.path.join(frame_folder, frame_name)
                frame = cv2.imread(frame_path)

                out.write(frame)

            out.release()

            clip_id += 1

    def run(self):

        videos = os.listdir(self.frames_dir)

        print(f"Found {len(videos)} videos")

        for video in tqdm(videos):

            video_path = os.path.join(self.frames_dir, video)

            if not os.path.isdir(video_path):
                continue

            self.generate_clips_for_video(video)


if __name__ == "__main__":

    frames_dir = "data/processed/frames"
    tracks_dir = "data/processed/tracks"
    output_dir = "data/processed/clips"

    generator = ClipGenerator(
        frames_dir,
        tracks_dir,
        output_dir
    )

    generator.run()