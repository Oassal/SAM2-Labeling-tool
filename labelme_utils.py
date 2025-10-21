import os
import json
import shutil
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def masks_paths(objects_names, parent_dir):
    masks_paths = {}
    for obj_name in objects_names:
        masks_paths[obj_name] = os.path.join(parent_dir, 'masks', obj_name)
    return masks_paths

def rearrange_contour_points_circular(contour):
    """
    Rearranges the contour points in a circular order.
    """
    # Calculate the centroid of the contour
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return contour
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Calculate the angle of each point relative to the centroid
    angles = np.arctan2(contour[:, 1] - cy, contour[:, 0] - cx)
    
    # Sort the points based on the angles
    sorted_indices = np.argsort(angles)
    sorted_contour = contour[sorted_indices]
    
    return sorted_contour

def rearrange_contour_points(contour):
    # Start with the first point as the reference point
    reference_point = contour[0]
    rearranged_contour = [reference_point]
    remaining_points = contour[1:].tolist()
    
    while remaining_points:
        # Find the closest point to the reference point
        distances = [np.linalg.norm(np.array(reference_point) - np.array(point)) for point in remaining_points]
        closest_point_index = np.argmin(distances)
        closest_point = remaining_points.pop(closest_point_index)
        
        # Add the closest point to the rearranged contour and update the reference point
        rearranged_contour.append(closest_point)
        reference_point = closest_point
    
    return np.array(rearranged_contour)


def create_labelme_annotation(image_path, masks):
    """
    Creates a LabelMe annotation for the given image and its corresponding masks.
    """
    image = Image.open(image_path)
    width, height = image.size
    
    shapes = []
    for class_id, mask_path in masks.items():

        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            
            # Apply erosion and dilation
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)
            mask  =cv2.threshold(mask,127,255,cv2.THRESH_BINARY)[1]
            if class_id == "pool":
                mask_wire = cv2.imread(masks['wire'],cv2.IMREAD_GRAYSCALE)
                mask_wire = cv2.threshold(mask_wire,127,255,cv2.THRESH_BINARY)[1]
                mask = mask - mask_wire
                mask[mask<100] = 0


            if class_id == "trajectory":
                mask_wire = cv2.imread(masks['wire'],cv2.IMREAD_GRAYSCALE)
                mask_wire = cv2.threshold(mask_wire,127,255,cv2.THRESH_BINARY)[1]

                mask_pool = cv2.imread(masks['pool'],cv2.IMREAD_GRAYSCALE)
                mask_pool = cv2.threshold(mask_wire,127,255,cv2.THRESH_BINARY)[1]
                # plt.imshow(mask)
                # plt.show()
                mask = mask - mask_pool - mask_wire
                mask[mask<100] = 0

            kernel2 = np.ones((25,25), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel2, iterations=1)
            # plt.imshow(mask)
            # plt.show()
            # Find contours and take only the biggest area
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                largest_contour = cv2.approxPolyDP(largest_contour, 0.01 * cv2.arcLength(largest_contour, True), True)
                largest_contour = largest_contour.squeeze()
                tmp_contour = []
                if class_id == 'trajectory':
                    for x,y in largest_contour:
                        if y<450: continue
                        else: tmp_contour.append([int(x),int(y)])
                    largest_contour = tmp_contour
                    points = tmp_contour
                else : points = largest_contour.tolist()
                shape = {
                    "label": class_id,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                shapes.append(shape)
    
    annotation = {
        "version": "4.5.6",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width
    }
    
    return annotation

def generate_labelme_dataset(images_dir,
                            objects_names,
                            ):
    """
    Generates a LabelMe dataset from images and their corresponding masks.
    objects_names: list of object names corresponding to mask subfolders.
    """
    masks_dir = os.path.join(images_dir, 'masks')
    masks_paths_dict = masks_paths(objects_names, images_dir)
    labelme_annotations_dir = images_dir+'/LabelMe_Dataset/'
    labelme_images_dir = images_dir+'/LabelMe_Dataset/'
    os.makedirs(labelme_annotations_dir, exist_ok=True)

    for image_file in os.listdir(images_dir):
        if image_file.endswith('.jpg'):
            image_name = os.path.splitext(image_file)[0]
            shutil.copy(os.path.join(images_dir, image_file), os.path.join(labelme_images_dir, image_file))
            # Create masks dictionary
            masks = {}
            for obj_name in objects_names:
                masks[obj_name] = os.path.join(masks_dir, obj_name, f"{int(image_name)-1}.png")
            
            # Create LabelMe annotation
            annotation = create_labelme_annotation(os.path.join(images_dir, image_file), masks)
            
            # Save annotation file
            with open(os.path.join(labelme_annotations_dir, f"{image_name}.json"), 'w') as annotation_file:
                json.dump(annotation, annotation_file, indent=4)


if __name__ == "__main__":
    images_dir = r'C:\Dataset\delete\vid\splits\20230830-103250961_split5'
    objects_names = ['[wire,', 'pool,', 'arc]']
    generate_labelme_dataset(images_dir, objects_names)
    print("Dataset has been transformed into LabelMe segmentation format.")
