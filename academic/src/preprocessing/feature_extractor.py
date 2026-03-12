import os
import cv2
import torch
import numpy as np
from tqdm import tqdm

from src.action.x3d_encoder import X3DEncoder


class FeatureExtractor:

    def __init__(self, clips_dir, output_dir, device="cuda"):

        self.clips_dir = clips_dir
        self.output_dir = output_dir
        self.device = device

        os.makedirs(self.output_dir, exist_ok=True)

        print("Loading X3D encoder...")
        self.encoder = X3DEncoder(device=self.device)

    def load_clip(self, clip_path):

        cap = cv2.VideoCapture(clip_path)

        frames = []

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame = cv2.resize(frame, (224, 224))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(frame)

        cap.release()

        frames = np.array(frames)

        frames = torch.from_numpy(frames).float() / 255.0
        frames = frames.permute(0, 3, 1, 2)   # T,C,H,W

        return frames

    def process_video(self, video_name):

        clip_folder = os.path.join(self.clips_dir, video_name)

        if not os.path.isdir(clip_folder):
            return

        output_folder = os.path.join(self.output_dir, video_name)
        os.makedirs(output_folder, exist_ok=True)

        clip_files = sorted(os.listdir(clip_folder))

        for clip in tqdm(clip_files):

            clip_path = os.path.join(clip_folder, clip)

            frames = self.load_clip(clip_path)

            frames = frames.unsqueeze(0).to(self.device)

            with torch.no_grad():

                features = self.encoder.encode(frames)

            feature_path = os.path.join(
                output_folder,
                clip.replace(".mp4", ".npy")
            )

            np.save(feature_path, features.cpu().numpy())

    def run(self):

        videos = os.listdir(self.clips_dir)

        print(f"Found {len(videos)} videos")

        for video in videos:

            print(f"Extracting features for {video}")

            self.process_video(video)


if __name__ == "__main__":

    clips_dir = "data/processed/clips"
    features_dir = "data/processed/features"

    extractor = FeatureExtractor(clips_dir, features_dir)

    extractor.run()