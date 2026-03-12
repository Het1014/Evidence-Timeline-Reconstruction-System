import os
import cv2
import argparse
from pathlib import Path
from tqdm import tqdm


def extract_frames(video_path, output_dir, target_fps=5):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error opening video: {video_path}")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)

    if original_fps == 0:
        original_fps = target_fps

    frame_interval = max(1, int(original_fps / target_fps))

    frame_count = 0
    saved_count = 0

    os.makedirs(output_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_name = f"frame_{saved_count:05d}.jpg"
            frame_path = os.path.join(output_dir, frame_name)

            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()

    print(f"Extracted {saved_count} frames from {video_path}")


def process_dataset(input_dir, output_dir, fps):

    categories = ["anomaly", "normal"]

    for category in categories:

        category_path = os.path.join(input_dir, category)

        if not os.path.exists(category_path):
            continue

        videos = list(Path(category_path).rglob("*.mp4"))

        print(f"\nProcessing {category} videos: {len(videos)} found")

        for video in tqdm(videos):

            video_path = str(video)
            video_name = Path(video_path).stem
            video_output_dir = os.path.join(output_dir, video_name)

            extract_frames(video_path, video_output_dir, fps)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/processed/frames",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=5,
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    process_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()