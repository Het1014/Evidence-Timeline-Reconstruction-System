# import os
# import cv2
# import json
# import numpy as np
# from tqdm import tqdm


# class VideoRanker:
#     def __init__(self, anomaly_root, output_json, sample_fps=1, resize_dim=(320, 240)):
#         self.anomaly_root = anomaly_root
#         self.output_json = output_json
#         self.sample_fps = sample_fps
#         self.resize_dim = resize_dim

#     def compute_motion_score(self, video_path):
#         cap = cv2.VideoCapture(video_path)

#         if not cap.isOpened():
#             return 0.0

#         fps = cap.get(cv2.CAP_PROP_FPS)
#         if fps <= 0:
#             fps = 25

#         frame_interval = int(fps // self.sample_fps)
#         prev_gray = None
#         motion_magnitudes = []

#         frame_idx = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if frame_idx % frame_interval != 0:
#                 frame_idx += 1
#                 continue

#             frame = cv2.resize(frame, self.resize_dim)
#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#             if prev_gray is not None:
#                 flow = cv2.calcOpticalFlowFarneback(
#                     prev_gray, gray,
#                     None,
#                     0.5, 3, 15, 3, 5, 1.2, 0
#                 )
#                 magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
#                 motion_magnitudes.append(magnitude)

#             prev_gray = gray
#             frame_idx += 1

#         cap.release()

#         if len(motion_magnitudes) == 0:
#             return 0.0

#         return float(np.mean(motion_magnitudes))

#     def rank_videos(self, top_k=4):
#         selected = {}

#         categories = sorted(os.listdir(self.anomaly_root))

#         for category in categories:
#             category_path = os.path.join(self.anomaly_root, category)

#             if not os.path.isdir(category_path):
#                 continue

#             print(f"\nProcessing category: {category}")

#             video_scores = []

#             videos = [
#                 v for v in os.listdir(category_path)
#                 if v.lower().endswith((".mp4", ".avi", ".mov"))
#             ]

#             for video in tqdm(videos):
#                 video_path = os.path.join(category_path, video)
#                 score = self.compute_motion_score(video_path)

#                 video_scores.append({
#                     "video": video,
#                     "path": video_path,
#                     "motion_score": score
#                 })

#             # Sort by motion score descending
#             video_scores = sorted(video_scores, key=lambda x: x["motion_score"], reverse=True)

#             selected[category] = video_scores[:top_k]

#         # Save selection
#         with open(self.output_json, "w") as f:
#             json.dump(selected, f, indent=4)

#         print(f"\nSelection saved to {self.output_json}")

#         return selected
    
# if __name__ == "__main__":
#     anomaly_root = "data/raw/"
#     output_json = "data/splits/selected_videos.json"

#     ranker = VideoRanker(anomaly_root, output_json)
#     ranker.rank_videos(top_k=4)


# --------------------------------------------------For other 4 normal videos--------------------------------------------------------
# import os
# import cv2
# import numpy as np
# from tqdm import tqdm

# NORMAL_DIR = "data/Normal"

# def analyze_video(video_path):
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         return None

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps <= 0:
#         fps = 25

#     frame_interval = int(fps)
#     prev_gray = None
#     motion_scores = []
#     brightness_vals = []
#     entropy_vals = []

#     frame_idx = 0

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         if frame_idx % frame_interval != 0:
#             frame_idx += 1
#             continue

#         frame = cv2.resize(frame, (320, 240))
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#         brightness_vals.append(np.mean(gray))

#         hist = cv2.calcHist([gray], [0], None, [256], [0,256])
#         hist_norm = hist / hist.sum()
#         entropy = -np.sum(hist_norm * np.log2(hist_norm + 1e-7))
#         entropy_vals.append(entropy)

#         if prev_gray is not None:
#             flow = cv2.calcOpticalFlowFarneback(
#                 prev_gray, gray,
#                 None,
#                 0.5, 3, 15, 3, 5, 1.2, 0
#             )
#             magnitude = np.mean(np.sqrt(flow[..., 0]**2 + flow[..., 1]**2))
#             motion_scores.append(magnitude)

#         prev_gray = gray
#         frame_idx += 1

#     cap.release()

#     if not motion_scores:
#         return None

#     return {
#         "motion": float(np.mean(motion_scores)),
#         "brightness": float(np.mean(brightness_vals)),
#         "entropy": float(np.mean(entropy_vals))
#     }


# results = []

# for file in tqdm(os.listdir(NORMAL_DIR)):
#     if file.endswith(".mp4"):
#         path = os.path.join(NORMAL_DIR, file)
#         stats = analyze_video(path)
#         if stats:
#             stats["video"] = file
#             results.append(stats)

# results = sorted(results, key=lambda x: x["motion"])

# print("\n--- Normal Video Stats (Sorted by Motion) ---\n")
# for r in results:
#     print(r)


# # --------------------------------------------------Update selected videos--------------------------------------------------------
# import json
# import os

# MANIFEST = "data/splits/selected_videos.json"
# NORMAL_DIR = "data/raw/Normal"

# # New normal videos to add
# new_videos = [
#     "Normal_Videos_798_x264.mp4",
#     "Normal_Videos_289_x264.mp4",
#     "Normal_Videos_641_x264.mp4",
#     "Normal_Videos_251_x264.mp4",
# ]

# with open(MANIFEST, "r") as f:
#     data = json.load(f)

# # Ensure Normal category exists
# if "Normal" not in data:
#     data["Normal"] = []

# existing_names = {v["video"] for v in data["Normal"]}

# for video in new_videos:
#     if video not in existing_names:
#         full_path = os.path.normpath(os.path.join("data/raw/Normal", video))

#         data["Normal"].append({
#             "video": video,
#             "path": full_path,
#             "motion_score": None  # Optional: leave None or recompute
#         })

# with open(MANIFEST, "w") as f:
#     json.dump(data, f, indent=4)

# print("Normal videos successfully updated.")