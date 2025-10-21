import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path
import random
import os
import cv2
import collections
import sys
from sam import generate_masks_video
from utils_folders import objects_dict_from_list


default_setOn = False
global default_setOfPoints
default_setOfPoints = {}
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def initialize_default_points(objects_names):
    global default_setOfPoints
    default_setOfPoints = {name: {"object_name": name, "object_points": [], "background_points": []} for name in objects_names}


def collect_points(object_name, image_path):
    """Collect object and background points from an image using mouse clicks."""
    global default_setOfPoints
    global default_setOn

    data = default_setOfPoints[object_name] if default_setOn else {
        "object_name": object_name,
        "object_points": [],
        "background_points": []
    }

    def onclick(event):
        if event.xdata is not None and event.ydata is not None:
            if event.button == 1:  # Left click
                data["object_points"].append((event.xdata, event.ydata))
                plt.scatter(event.xdata, event.ydata, color='red')  # Visually mark the class object point
            elif event.button == 3:  # Right click
                data["background_points"].append((event.xdata, event.ydata))
                plt.scatter(event.xdata, event.ydata, color='blue')  # Visually mark the background point
            plt.draw()

    def onpress(event):
        sys.stdout.flush()
        if event.key == 'enter':
            plt.close()
        if event.key == 'd':
                data["object_points"] = []
                data["background_points"] = []
                for scatter in plt.gca().collections:
                    scatter.remove()
                plt.draw()
        if event.key == 'n':
            # Define a default set of points.
            #TODO make the generation of default dict (0) automatic
            global default_setOfPoints
            global default_setOn
            default_setOn = True
            default_setOfPoints[object_name] = data
    
    # Load and display the image
    img = plt.imread(image_path)
    plt.imshow(img)
    if default_setOn and data["object_points"] and data["background_points"]:
        x_obj, y_obj = zip(*data["object_points"])
        x_bg, y_bg = zip(*data["background_points"])
        plt.scatter(x_obj, y_obj, color='red', marker='o')
        plt.scatter(x_bg, y_bg, color='blue', marker='o')
    manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.title(f"Object: {object_name} - Click to collect points (Left: Object, Right: Background)", color='red')
    cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
    cid_2 = plt.gcf().canvas.mpl_connect('key_press_event',onpress)
    plt.show()
    return data

def label_one_point(image_path,
                    objects_names=None,
                    ):
    """Label one image by collecting points for each object."""
    image_path = str(image_path)
    all_data = []

    # Collect points for each object
    if objects_names == None:
        raise ValueError("Object names must be provided.")

    for object_name in objects_names:
        object_data = collect_points(object_name, image_path)
        all_data.append(object_data)

    # Save all points to a JSON file
    p = Path(image_path)
    labels_dir = (p.parent / "Labels")
    labels_dir.mkdir(parents=True, exist_ok=True)
    json_path = str(labels_dir / p.with_suffix('.json').name)

    with open(json_path, 'w') as f:
        json.dump(all_data, f, indent=4)
    print("Points saved to " + json_path)


def label_dir(imgs_dir = None,
              sequ_length = 100,
              video_dir = None,
              objects_names = [],
              Nb_Random_Images = 2
              ):
    '''
    Take a video, or a list of images ??
    if a video, split it
    if only images --> sample them and start labeling
    '''
    extensions = ['.jpg','.jpeg','.png']
    
    imgs_list = []
    if len(objects_names)== 0:
        num_objects = int(input("Enter the number of objects: "))
        for i in range(num_objects):
            object_name = input(f"Enter the name for object {i + 1}: ")
            objects_names.append(object_name)
    if not imgs_dir is None:
        imgs_dir = Path(imgs_dir)
    elif video_dir is not None:
        splits_dirs = split_videos_into_frames(video_dir,sequ_length)
    else:
        imgs_dir = Path(input("Enter the images directory path: "))

    for img in imgs_dir.rglob("*"):
        if img.suffix in extensions:
            imgs_list.append(str(img))


    # Take first image, middle image and 3 random images
    images_to_annotate = []
    images_to_annotate_idx=[]
    for i in range(Nb_Random_Images):
        while True:
            rand_index = int(random.random() * sequ_length)
            if rand_index != 0 and rand_index != sequ_length-1 and rand_index != sequ_length/2: 
                break
        images_to_annotate_idx.append(rand_index)
    images_to_annotate_idx.append(0)
    images_to_annotate_idx.append(sequ_length-1)
    images_to_annotate_idx.append(sequ_length//2)
    images_to_annotate_idx.sort()            
    images_to_annotate = [imgs_list[idx] for idx in images_to_annotate_idx]
    for img in images_to_annotate:
        label_one_point(img,objects_names)

def split_videos_into_frames(directory, frames_per_split):
    # Check if the directory exists
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    
    # Possible video file extensions
    video_extensions = ('.webm', '.avi', '.mp4')

    output_dir = os.path.join(directory, 'splits')
    os.makedirs(output_dir, exist_ok=True)
    splits_dirs = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith(video_extensions):
                video_path = os.path.join(root, filename)
                vid_cap = cv2.VideoCapture(video_path)
                total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                split_count = total_frames // frames_per_split + (1 if total_frames % frames_per_split > 0 else 0)
                
                
                split_index = 0
                while True:
                    # Create folder for this split
                    split_index+=1
                    split_folder = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_split{split_index}')
                    os.makedirs(split_folder, exist_ok=True)
                    current_frame = 0
                    splits_dirs.append(split_folder)
                    for frame_index in range(frames_per_split):
                        ret, frame = vid_cap.read()
                        if not ret:
                            break
                        
                        # Save the frame
                        frame_filename = os.path.join(split_folder, f'{(current_frame + 1):06d}.jpg')
                        cv2.imwrite(frame_filename, frame)
                        current_frame += 1
                    if not ret: break                    

                vid_cap.release()
    return splits_dirs


def read_files_for_SAM2(parent_directory,
                        objects_names ,
                        sam2_checkpoint="sam2_hiera_large.pt",
                        model_cfg="sam2_hiera_l.yaml"
                        ):
    if not os.path.exists(parent_directory):
        raise FileNotFoundError(f"The directory '{parent_directory}' does not exist.")
    
    labels_found = False
    labelFolders_list = []

    for root, dirs, files in os.walk(parent_directory):
        # Check if 'Labels' folder exists in the current directory
        # 'Labels' folder is created during the labeling process, in previous functions
        if "Labels" in dirs:
            labels_found = True
            labelFolders_list.append(root)
            print(f"'Labels' folder found in: {root}")

    #TODO make the objects dict automatic
    objects_dict = objects_dict_from_list(objects_names)

    if not labels_found:
        raise FileNotFoundError("No 'Labels' folder found in any subdirectory.")

    for root in labelFolders_list:
        proot = Path(root)/'Labels'
        ann_frame_indexes = []
        ann_obj = collections.defaultdict(list)
        for jsonFile in proot.rglob("*.json"):
            # jsonFile = jsonFile/"Labels"
            frame_idx = int(os.path.splitext(jsonFile.name)[0])-1
            ann_frame_indexes.append(frame_idx)
            with open(jsonFile,'r') as file:
                data = json.load(file)
            
            for obj in data:
                ann_obj_id = objects_dict[obj['object_name']]
                obj_points = [point for point in obj['object_points']]
                flags = [1 for point in  obj['object_points']]

                bg_points = [point for point in obj['background_points']]
                flags+=[0 for point in obj['background_points']]
                points = obj_points+bg_points
                ann_obj[frame_idx].append({'ann_obj_id':ann_obj_id, 'points':points, 'labels':flags})
        generate_masks_video(root,sam2_checkpoint,model_cfg,objects_dict,ann_frame_indexes,ann_obj)