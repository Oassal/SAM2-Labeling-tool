# SAM2-Labeling-tool

## Overview

This repository provides a semi-automatic labeling pipeline for video and image datasets using the SAM2 segmentation model. It enables users to:
- Split videos into frames
- Annotate images interactively with foreground/background points
- Generate segmentation masks using SAM2
- Export datasets in LabelMe format for downstream tasks

## Features
- **Interactive annotation:** Collect object/background points via mouse clicks
- **Batch processing:** Split videos and label images in bulk
- **Automatic mask generation:** Use SAM2 to create segmentation masks
- **LabelMe export:** Convert masks and annotations to LabelMe JSON format

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/Oassal/SAM2-Labeling-tool.git
   ```
2. Install dependencies (recommended: use Anaconda):
   ```
   conda create -n SAM2 python=3.10
   conda activate SAM2
   pip install -r requirements.txt
   ```
   - Required packages: numpy, matplotlib, opencv-python, Pillow, torch
   - You also need the SAM2 model weights and config files (see below)

## Usage

### 1. Split Videos into Frames
Run the main pipeline with your video directory:
```bash
python main.py --video_dir "path/to/videos" --frames_per_split 100 --objects_names object1 object2 object3 ...
```
- --frames_per_split is the sampling value for each sequence, this value depends on the used GPU
- This will split each video into folders of frames.

### 2. Annotate Images
The script will prompt you to annotate selected images in each split folder. Use mouse clicks:
- **Left click:** Add object point
- **Right click:** Add background point
- **Press 'n':** Set current points as default for the object
- **Press 'd':** Clear all points
- **Press 'enter':** Save and close annotation

### 3. Generate Masks
After annotation, the pipeline will automatically generate segmentation masks using SAM2 for each split folder.

### 4. Export to LabelMe Format (Optional)
Add the `--labelme` flag to export the dataset in LabelMe format:
```bash
python main.py --video_dir "C:\path\to\videos" --frames_per_split 100 --objects_names object1 object2 object3 ... --labelme
```

## File Structure
- `main.py`: Main pipeline script
- `utils_labeling.py`: Annotation and mask generation utilities
- `labelme_utils.py` / `labelme_dataset.py`: LabelMe export utilities
- `utils_folders.py`: Folder management utilities
- `sam.py`: SAM2 mask generation logic

## Model Files
- Place your SAM2 model weights and config files in the project directory:
  - `sam2_hiera_large.pt`
  - `sam2_hiera_l.yaml`

## Example Workflow
1. Place your videos in a folder
2. Run the main script with appropriate arguments
3. Annotate images as prompted
4. Masks and LabelMe datasets will be generated in each split folder

## To be done
- Possibility of using other models
- Possibility of continuing labeling where you left off
...

## License
MIT License