import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor
from pathlib import Path

# Device selection
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


######################## helper functions ###########################

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


########################## end of helper functions ##################

def generate_masks_video(imgs_dir,
                         sam2_checkpoint,
                         model_cfg,
                         objects_dict:dict,
                         ann_frame_indexes:list,
                         ann_obj:dict):
    '''
    A function that takes as input the directory of images
    and generates the corresponding masks.
    The directory of the masks is masks of the same folder,
    with subfolders for each mask object, e.g. one folder for pool...etc
    '''
    predictor = build_sam2_video_predictor(model_cfg,sam2_checkpoint,device=device)
    masks_dir = Path(imgs_dir)/"masks"
    masks_dir.mkdir(parents=True,exist_ok=True)
    objects_list = []
    for obj_key in objects_dict.keys():
        obj_dir = masks_dir/obj_key
        obj_dir.mkdir(parents=True,exist_ok=True)
        objects_list.append(obj_key)
    # Scanning all frame names, pngs, jpegs and stuff
    frame_names = [
        p for p in os.listdir(imgs_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG",".png",".PNG"]
    ]

    frame_names.sort(key =lambda p: int(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path = imgs_dir)
    predictor.reset_state(inference_state)
    for ann_frame_idx in ann_frame_indexes:
        frame_annotation = ann_obj[ann_frame_idx]
        for obj in frame_annotation:
            ann_obj_id = obj['ann_obj_id']
            points = obj['points']
            labels = obj['labels']
            if points:
                _ = predictor.add_new_points_or_box(
                    inference_state = inference_state,
                    frame_idx = int(ann_frame_idx),
                    obj_id = ann_obj_id,
                    points= points,
                    labels= labels
                )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }
    for out_obj_id, out_masks in video_segments.items():
        for obj, mask in out_masks.items():
            a = np.array(mask.squeeze(), dtype= np.uint8)
            plt.imsave(str(masks_dir/f'{objects_list[obj]}/{out_obj_id}.png'),a)