import os
import torch
import numpy as np
from src.detection.detection_utils import fuse_anomaly_scores
from src.modeling.transformer_model import TemporalTransformer
from src.preprocessing.track_serializer import build_detection_features
import random

FEATURES_DIR = "data/processed/features"
CHECKPOINT_DIR = "models/checkpoints"

def smooth(preds, window=5):
    smoothed = np.convolve(preds, np.ones(window)/window, mode='same')
    return (smoothed > 0.5).astype(int)


def load_video_features(video_name):

    video_path = os.path.join(FEATURES_DIR, video_name)

    clips = sorted(os.listdir(video_path))

    features = []

    for clip in clips:

        if clip.endswith(".npy"):

            f = np.load(os.path.join(video_path, clip))
            features.append(f.squeeze())

    features = np.stack(features)

    return torch.tensor(features, dtype=torch.float32).unsqueeze(0)


def run_inference(model_path, video_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TemporalTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    model = model.to(device)
    model.eval()

    features = load_video_features(video_name).to(device)

    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=-1)
        anomaly_scores = probs[0,:,1].cpu().numpy()
        detections_per_clip = build_detection_features(video_name)
        min_len = min(len(anomaly_scores), len(detections_per_clip))
        anomaly_scores = anomaly_scores[:min_len]
        detections_per_clip = detections_per_clip[:min_len]
        fused_scores = fuse_anomaly_scores(anomaly_scores, detections_per_clip)
        threshold = fused_scores.mean()
        preds = (fused_scores > threshold).astype(int)

    print("\nCheckpoint:", model_path)
    print("Video:", video_name)
    print("Predictions shape:", preds.shape)
    print("Predicted classes (first 50 clips):")
    print(preds[:50])
    print("Fused scores:", fused_scores)
    print("Anomaly score threshold:", threshold)
    print("Max anomaly score:", anomaly_scores.max())


if __name__ == "__main__":

    videos = [
        v for v in os.listdir(FEATURES_DIR)
        if os.path.isdir(os.path.join(FEATURES_DIR, v))
    ]

    video_name = random.choice(videos)
    checkpoints = [
        "models/checkpoints/epoch_5.pth",
        "models/checkpoints/epoch_10.pth",
        "models/checkpoints/best_model.pth"
    ]

    for ckpt in checkpoints:

        if os.path.exists(ckpt):

            run_inference(ckpt, video_name)