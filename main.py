from utils_labeling import initialize_default_points, split_videos_into_frames, label_dir, read_files_for_SAM2
from utils_folders import list_subfolders
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Auto Labeling with SAM2")
parser.add_argument("--video_dir", type=str, help="Path to the video directory")
parser.add_argument("--frames_per_split", type=int, default=100, help="Number of frames per split")
parser.add_argument("--objects_names", type=str, nargs='+', help="List of object names")
parser.add_argument("--labelme", action='store_true', help="Flag to create LabelMe dataset")
args = parser.parse_args()

def main():
    initialize_default_points(args.objects_names)
    # Step 1: Split videos into frames
    video_dir = args.video_dir
    frames_per_split = args.frames_per_split
    split_dirs = split_videos_into_frames(video_dir, frames_per_split)
    print("Video splits created:", split_dirs)

    # Step 2: Label images in each split directory
    objects_names = args.objects_names
    for split_dir in split_dirs:
        img_count = len([f for f in Path(split_dir).rglob("*.jpg")])
        label_dir(imgs_dir=split_dir, sequ_length=img_count, objects_names=objects_names)
        print(f"Labeled images in {split_dir}")
    
    # split_dirs = list_subfolders(r'C:\Dataset\delete\vid\splits')
    # Step 3: Generate masks using SAM2
    for split_dir in split_dirs:
        read_files_for_SAM2(split_dir, objects_names=args.objects_names)
        print(f"Masks generated for {split_dir}")

    # Step 4: Optionally create LabelMe dataset
    if args.labelme:
        from labelme_utils import generate_labelme_dataset
        for split_dir in split_dirs:
            generate_labelme_dataset(split_dir, objects_names=args.objects_names)
            print(f"LabelMe dataset created for {split_dir}")

if __name__ == "__main__":
    main()