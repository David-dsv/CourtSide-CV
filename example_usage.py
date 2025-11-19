"""
Example Usage Scripts
Quick examples for using the tennis analysis pipeline
"""

import sys
from pathlib import Path

# Example 1: Complete Pipeline
def example_full_pipeline():
    """Run complete end-to-end pipeline"""
    from main import TennisAnalysisPipeline

    print("Example 1: Full Pipeline")
    print("=" * 60)

    # Initialize pipeline
    pipeline = TennisAnalysisPipeline("config.yaml")

    # Run on video
    results = pipeline.run(video_path="data/input/tennis_match.mp4")

    print(f"Detected {results['summary']['total_actions']} actions")
    print(f"Action breakdown: {results['summary']['action_counts']}")


# Example 2: Detection Only
def example_detection_only():
    """Run YOLO detection only"""
    from models.yolo_detector import TennisYOLODetector
    from inference.detect import detect_video

    print("\nExample 2: Detection Only")
    print("=" * 60)

    # Initialize detector
    detector = TennisYOLODetector(
        model_version="yolov8",
        model_size="m",
        device="cuda"
    )

    # Run detection
    detections = detect_video(
        video_path="data/input/tennis_match.mp4",
        detector=detector,
        output_video="output/detections.mp4",
        output_json="output/detections.json"
    )

    print(f"Processed {len(detections)} frames")


# Example 3: Custom Training
def example_train_action_classifier():
    """Train custom action classifier"""
    import torch
    from torch.utils.data import DataLoader
    from models.action_classifier import TennisActionClassifier
    from training.dataset import TennisActionDataset
    from training.transforms import get_train_transforms

    print("\nExample 3: Train Action Classifier")
    print("=" * 60)

    # Create dataset
    train_transform = get_train_transforms(spatial_size=224, clip_length=32)

    train_dataset = TennisActionDataset(
        data_root="data/",
        split="train",
        clip_length=32,
        spatial_size=224,
        transform=train_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4
    )

    # Create model
    model = TennisActionClassifier(
        model_type="x3d",
        model_size="m",
        num_classes=6,
        pretrained=True
    ).cuda()

    # Training loop (simplified)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):  # Just 5 epochs for example
        total_loss = 0
        for clips, labels in train_loader:
            clips = clips.cuda()
            labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    # Save model
    torch.save(model.state_dict(), "weights/custom_action_classifier.pth")
    print("Model saved!")


# Example 4: Batch Processing
def example_batch_processing():
    """Process multiple videos"""
    from main import TennisAnalysisPipeline

    print("\nExample 4: Batch Processing")
    print("=" * 60)

    pipeline = TennisAnalysisPipeline("config.yaml")

    video_dir = Path("data/input/")
    video_files = list(video_dir.glob("*.mp4"))

    print(f"Found {len(video_files)} videos to process")

    for i, video_file in enumerate(video_files, 1):
        print(f"\nProcessing {i}/{len(video_files)}: {video_file.name}")

        # Set output directory for this video
        output_dir = Path("output") / video_file.stem
        pipeline.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run pipeline
        results = pipeline.run(video_path=str(video_file))

        print(f"  Detected {results['summary']['total_actions']} actions")


# Example 5: Real-time Webcam
def example_realtime_webcam():
    """Real-time detection from webcam"""
    import cv2
    import numpy as np
    from models.yolo_detector import TennisYOLODetector
    from utils.drawing import draw_bboxes

    print("\nExample 5: Real-time Webcam")
    print("=" * 60)
    print("Press 'q' to quit")

    # Initialize detector
    detector = TennisYOLODetector(
        model_version="yolov8",
        model_size="n",  # Use nano for speed
        device="cuda"
    )

    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect
        detections = detector.detect(frame)

        # Draw
        if len(detections["boxes"]) > 0:
            labels = [
                f"{name} {score:.2f}"
                for name, score in zip(detections["class_names"], detections["scores"])
            ]
            frame = draw_bboxes(frame, detections["boxes"], labels=labels)

        # Display
        cv2.imshow("Tennis Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Example 6: Extract Specific Actions
def example_extract_specific_actions():
    """Extract all serves from a match"""
    from main import TennisAnalysisPipeline
    from inference.extract_clips import extract_action_clips

    print("\nExample 6: Extract Specific Actions")
    print("=" * 60)

    # Run pipeline
    pipeline = TennisAnalysisPipeline("config.yaml")
    results = pipeline.run(video_path="data/input/tennis_match.mp4")

    # Filter for serves only
    serves = [
        action for action in results["actions"]
        if action["action"] == "serve"
    ]

    print(f"Found {len(serves)} serves")

    # Extract serve clips
    if serves:
        extract_action_clips(
            "data/input/tennis_match.mp4",
            serves,
            output_dir="output/serves/",
            margin_frames=20
        )

        print(f"Saved serve clips to output/serves/")


# Example 7: Analyze Specific Player
def example_analyze_player():
    """Analyze actions of a specific tracked player"""
    from main import TennisAnalysisPipeline

    print("\nExample 7: Analyze Specific Player")
    print("=" * 60)

    pipeline = TennisAnalysisPipeline("config.yaml")
    results = pipeline.run(video_path="data/input/tennis_match.mp4")

    # Get actions for track_id 1 (first player)
    player1_actions = [
        action for action in results["actions"]
        if action.get("track_id") == 1
    ]

    print(f"Player 1 actions: {len(player1_actions)}")

    # Count action types
    action_counts = {}
    for action in player1_actions:
        action_type = action["action"]
        action_counts[action_type] = action_counts.get(action_type, 0) + 1

    print("Action breakdown:")
    for action_type, count in sorted(action_counts.items()):
        print(f"  {action_type}: {count}")


# Example 8: Export Results to Different Formats
def example_export_results():
    """Export results in various formats"""
    from main import TennisAnalysisPipeline
    from utils.file_utils import save_json, save_results_csv
    import pandas as pd

    print("\nExample 8: Export Results")
    print("=" * 60)

    pipeline = TennisAnalysisPipeline("config.yaml")
    results = pipeline.run(video_path="data/input/tennis_match.mp4")

    # Export to JSON
    save_json(results["actions"], "output/actions.json")
    print("Saved JSON: output/actions.json")

    # Export to CSV
    save_results_csv(
        results["actions"],
        "output/actions.csv",
        columns=["frame_id", "timestamp", "action", "confidence", "track_id"]
    )
    print("Saved CSV: output/actions.csv")

    # Export to Excel
    df = pd.DataFrame(results["actions"])
    df.to_excel("output/actions.xlsx", index=False)
    print("Saved Excel: output/actions.xlsx")

    # Export summary statistics
    summary = {
        "total_frames": results["summary"]["total_frames"],
        "total_actions": results["summary"]["total_actions"],
        "action_counts": results["summary"]["action_counts"],
        "actions_per_frame": results["summary"]["total_actions"] / results["summary"]["total_frames"]
    }

    save_json(summary, "output/summary.json")
    print("Saved summary: output/summary.json")


def main():
    """Run all examples"""
    print("Tennis Action Analysis - Example Usage")
    print("=" * 60)

    examples = [
        ("Full Pipeline", example_full_pipeline),
        ("Detection Only", example_detection_only),
        ("Batch Processing", example_batch_processing),
        ("Extract Specific Actions", example_extract_specific_actions),
        ("Analyze Player", example_analyze_player),
        ("Export Results", example_export_results),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"{i}. {name}")

    print("\nNote: Some examples require video files in data/input/")
    print("Run individual examples by uncommenting them below")

    # Uncomment to run specific examples:
    # example_full_pipeline()
    # example_detection_only()
    # example_batch_processing()
    # example_extract_specific_actions()
    # example_analyze_player()
    # example_export_results()


if __name__ == "__main__":
    main()
