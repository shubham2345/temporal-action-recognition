#!/usr/bin/env python
"""
Main module for training, inference, and evaluation of the Temporal Localization Model.

This module:
    - Loads the THUMOS dataset from .npy features and JSON annotations.
    - Splits the dataset into training, validation, and test sets.
    - Trains the Temporal Localization Model using AMP for mixed precision.
    - Runs inference on the test set.
    - Post-processes predictions into merged segments.
    - Computes evaluation metrics (mAP) using IoU.

Usage:
    python main.py
"""

import os
import json
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from evaluation import compute_map, post_process_predictions
from model import TemporalLocalizationModel


def load_json_annotations(json_path):
    """
    Load JSON annotations and convert segment times to floats.

    Args:
        json_path (str): Path to the JSON file.

    Returns:
        dict: Loaded annotations.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for _, info in data.items():
        for ann in info["annotations"]:
            ann["segment"] = list(map(float, ann["segment"]))
    return data


def build_label_mapping(annotations):
    """
    Build a mapping from label strings to numeric IDs, reserving 0 for background.

    Args:
        annotations (dict): Annotations dictionary.

    Returns:
        dict: Label mapping.
    """
    unique_actions = set()
    for _, info in annotations.items():
        for ann in info["annotations"]:
            unique_actions.add(ann["label"])
    mapping = {"background": 0}
    for i, action in enumerate(sorted(unique_actions), start=1):
        mapping[action] = i
    return mapping


def filter_samples_by_video_ids(dataset, video_id_set):
    """
    Filter dataset samples based on video IDs.

    Args:
        dataset (Dataset): The dataset instance.
        video_id_set (set): Set of video IDs to include.

    Returns:
        list: List of indices.
    """
    return [i for i, (video_id, _) in enumerate(dataset.samples)
            if video_id in video_id_set]


def build_ground_truth_segments(annotations, label_mapping):
    """
    Build ground truth segments dictionary.

    Format: {class_label: {video_id: [(start_time, end_time), ...], ...}, ...}

    Args:
        annotations (dict): A dictionary containing video annotations.
            Expected format:
                {
                    video_id: {
                        "annotations": [
                            {"segment": [start_time, end_time], "label": "ActionName"},
                            ...
                        ]
                    },
                    ...
                }
        label_mapping (dict): A mapping from label string to a numeric label.
            Background is assumed to be mapped to 0 and will be skipped.

    Returns:
        dict: Ground truth segments.
    """
    ground_truth = {}
    for video_id, info in annotations.items():
        for ann in info["annotations"]:
            numeric_label = label_mapping.get(ann["label"], 0)
            if numeric_label == 0:
                continue  # Skip background
            seg = tuple(map(float, ann["segment"]))
            if numeric_label not in ground_truth:
                ground_truth[numeric_label] = {}
            if video_id not in ground_truth[numeric_label]:
                ground_truth[numeric_label][video_id] = []
            ground_truth[numeric_label][video_id].append(seg)
    return ground_truth


# pylint: disable=too-many-instance-attributes
class ThumosSlidingWindowDataset(Dataset):
    """
    Dataset for sliding window samples from THUMOS .npy features.
    """
    def __init__(self, anno_path, feature_dir, window_size=16, window_stride=8, fps=25.0):
        """
        Initialize the dataset.

        Args:
            anno_path (str): Path to JSON annotation file.
            feature_dir (str): Directory with .npy feature files.
            window_size (int): Frames per window.
            window_stride (int): Stride for sliding windows.
            fps (float): Frames per second.
        """
        self.feature_dir = feature_dir
        self.annotations = load_json_annotations(anno_path)
        self.window_size = window_size
        self.window_stride = window_stride
        self.fps = fps

        self.label_mapping = build_label_mapping(self.annotations)
        print("Label Mapping:", self.label_mapping)

        self.features = {}
        self.samples = []
        skipped_videos = []
        for video_id in self.annotations.keys():
            feature_path = os.path.join(self.feature_dir, f"{video_id}.npy")
            try:
                features = np.load(feature_path)
                self.features[video_id] = features
                num_frames = features.shape[0]
                for start_frame in range(0, num_frames - window_size + 1, window_stride):
                    self.samples.append((video_id, start_frame))
            except FileNotFoundError:
                skipped_videos.append(video_id)
                print(f"Skipped {video_id} due to missing feature files")
                continue
        print(f"Loaded {len(self.samples)} samples from "
              f"{len(self.annotations) - len(skipped_videos)} videos")
        if skipped_videos:
            print(f"Skipped {len(skipped_videos)} videos due to missing feature files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_id, start_frame = self.samples[idx]
        features = self.features[video_id]
        window_features = features[start_frame:start_frame + self.window_size, :]
        window_start_time = start_frame / self.fps
        window_end_time = (start_frame + self.window_size) / self.fps
        video_info = self.annotations[video_id]
        best_overlap = 0.0
        best_action = "background"
        for ann in video_info["annotations"]:
            seg_start, seg_end = ann["segment"]
            overlap = min(window_end_time, seg_end) - max(window_start_time, seg_start)
            if overlap > best_overlap:
                best_overlap = overlap
                best_action = ann["label"]
        numeric_label = self.label_mapping.get(best_action, 0)
        return {
            "video_id": video_id,
            "window_features": torch.tensor(window_features, dtype=torch.float32),
            "label": torch.tensor(numeric_label, dtype=torch.long),
            "action": best_action,
            "window_start_time": window_start_time,
            "window_end_time": window_end_time,
        }


def main():
    """
    Main function to initialize and process the THUMOS dataset.
    
    This function sets up the data pipeline for the THUMOS action recognition dataset:
    - Loads action annotations from a JSON file
    - Processes pre-extracted features from the specified directory
    - Creates a sliding window dataset for temporal action localization
    
    Paths:
        - anno_path: Path to the THUMOS annotation JSON file
        - feature_dir: Directory containing pre-extracted features
    
    Parameters:
        None
    
    Returns:
        None
    
    Dataset Configuration:
        - Window size: 16 frames
        - Window stride: 8 frames
        - Frame rate: 25.0 FPS
    """
    # Set paths
    anno_path = "thumos_anno_action.json"
    feature_dir = "thumos_features/"
    dataset = ThumosSlidingWindowDataset(anno_path, feature_dir,
                                          window_size=16, window_stride=8, fps=25.0)
    print("Full Dataset Samples:", len(dataset))

    all_video_ids = set(dataset.annotations.keys()) - {"video_test_0001292"}
    train_val_video_ids = [vid for vid in all_video_ids if "validation" in vid]
    print("Train+Val Videos:", len(train_val_video_ids))
    train_val_video_ids = train_val_video_ids[:150]
    print("Reduced Train+Val Videos:", len(train_val_video_ids))
    test_video_ids = [vid for vid in all_video_ids if "test" in vid]
    print("Test Videos:", len(test_video_ids))

    train_video_ids, val_video_ids = train_test_split(train_val_video_ids,
                                                       test_size=0.2,
                                                       random_state=42)
    print("Train Videos:", len(train_video_ids))
    print("Validation Videos:", len(val_video_ids))

    train_indices = filter_samples_by_video_ids(dataset, set(train_video_ids))
    val_indices = filter_samples_by_video_ids(dataset, set(val_video_ids))
    test_indices = filter_samples_by_video_ids(dataset, set(test_video_ids))
    print("Train Samples:", len(train_indices))
    print("Validation Samples:", len(val_indices))
    print("Test Samples:", len(test_indices))

    batch_size = 32
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=4, pin_memory=True)

    num_classes = len(dataset.label_mapping)
    print("Number of Classes:", num_classes)
    model = TemporalLocalizationModel(feature_dim=768, hidden_dim=256,
                                      num_classes=num_classes)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    epochs = 10
    best_val_acc = 0.0
    best_model_path = "best_model.pth"

    # TRAINING LOOP
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_total = 0
        train_correct = 0
        for batch in train_loader:
            inputs = batch["window_features"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        train_loss = running_loss / train_total
        train_acc = train_correct / train_total

        model.eval()
        val_running_loss = 0.0
        val_total = 0
        val_correct = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch["window_features"].to(device)
                labels = batch["label"].to(device)
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_loss = val_running_loss / val_total
        val_acc = val_correct / val_total
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"Saved best model with Val Acc: {best_val_acc:.4f}")

    model.load_state_dict(torch.load(best_model_path))
    print("Best model loaded for inference.")

    # INFERENCE & POST-PROCESSING
    model.eval()
    all_predictions = []
    inv_label_mapping = {v: k for k, v in dataset.label_mapping.items()}
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["window_features"].to(device)
            with autocast():
                outputs = model(inputs)
                probs = nn.functional.softmax(outputs, dim=1)
                conf, pred_labels = torch.max(probs, dim=1)
            batch_video_ids = batch["video_id"]
            batch_start_times = batch["window_start_time"]
            batch_end_times = batch["window_end_time"]
            batch_pred_labels = pred_labels.cpu().numpy()
            batch_confidences = conf.cpu().numpy()
            for i, _ in enumerate(batch_video_ids):
                pred_dict = {
                    "video_id": batch_video_ids[i],
                    "window_start_time": batch_start_times[i],
                    "window_end_time": batch_end_times[i],
                    "label": int(batch_pred_labels[i]),
                    "action": inv_label_mapping[int(batch_pred_labels[i])],
                    "confidence": float(batch_confidences[i]),
                }
                all_predictions.append(pred_dict)
    print("Total predictions:", len(all_predictions))
    merged_segments = post_process_predictions(all_predictions, min_gap=5.0)
    print("Merged Segments:")
    for vid, segments in merged_segments.items():
        print(f"Video ID: {vid}")
        for seg in segments:
            print(seg)

    all_ground_truths = build_ground_truth_segments(dataset.annotations,
                                                      dataset.label_mapping)
    all_preds_list = []
    for vid, segs in merged_segments.items():
        for seg in segs:
            all_preds_list.append({
                "video_id": vid,
                "start_time": seg["start_time"],
                "end_time": seg["end_time"],
                "label": seg["label"],
                "confidence": seg.get("confidence", 1.0),
                "action": seg["action"],
            })
    map_score, ap_dict = compute_map(all_preds_list, all_ground_truths, iou_threshold=0.5)
    print("mAP:", map_score)
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(f"mAP: {map_score}\n")
        f.write(f"AP per class: {ap_dict}\n")


if __name__ == "__main__":
    main()
