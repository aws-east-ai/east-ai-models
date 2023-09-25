import os
import supervision as sv
import argparse
from functools import partial
import cv2
import requests
from io import BytesIO
import io
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
from torchvision.ops import box_convert
from groundingdino.models import build_model
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict
from groundingdino.util.inference import annotate, predict
import groundingdino.datasets.transforms as T
from huggingface_hub import hf_hub_download
import torch
import sys
from segment_anything import sam_model_registry, SamPredictor
import json
import boto3
import uuid
import wget


def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file) 
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model    


def generate_masks_with_grounding(image_source, boxes):
    box_list = []
    h, w, _ = image_source.shape
    boxes_unnorm = boxes * torch.Tensor([w, h, w, h])
    boxes_xyxy = box_convert(boxes=boxes_unnorm, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    mask = np.zeros_like(image_source)
    for box in boxes_xyxy:
        x0, y0, x1, y1 = box
        box_list.append(np.array([int(x0),int(y0),int(x1),int(y1)]))
        mask[int(y0):int(y1), int(x0):int(x1), :] = 255
    return mask, box_list


def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))  


def input_fn(request_body, request_content_type):
    print(f"=================input_fn=================\n{request_content_type}\n{request_body}")
    input_data = json.loads(request_body)
    return input_data


def model_fn(model_dir):
    print("=================model_fn=================")
    # Use this command for evaluate the Grounding DINO model
    # Or you can download the model by yourself
    ckpt_repo_id = "ShilongLiu/GroundingDINO"
    ckpt_filenmae = "groundingdino_swint_ogc.pth"
    ckpt_config_filename = "GroundingDINO_SwinT_OGC.cfg.py"
    model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename)
    
    ## Loading sam model
    sam_url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    # sam_checkpoint = "/opt/ml/model/sam_vit_h_4b8939.pth"
    sam_checkpoint = "/tmp/sam_vit_h_4b8939.pth"
    wget.download(sam_url, sam_checkpoint)
    model_type = "vit_h"
    device = "cuda"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    model_dic = {'dino': model, 'sam': sam}
    print("=================model load complete=================")
    return model_dic


def predict_fn(input_data, model):
    print("=================Dino detect start=================")
    dir_lst = input_data['input_image'].split('/')
    s3_client = boto3.client('s3')
    s3_response_object = s3_client.get_object(Bucket=dir_lst[2], Key='/'.join(dir_lst[3:]))
    img_bytes = s3_response_object['Body'].read()
    
    try:
        TEXT_PROMPT = input_data['prompt']
        BOX_TRESHOLD = 0.5
        TEXT_TRESHOLD = 0.5

        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = np.asarray(image_source)
        image_transformed, _ = transform(image_source, None)

        boxes, logits, phrases = predict(
            model=model['dino'], 
            image=image_transformed, 
            caption=TEXT_PROMPT, 
            box_threshold=BOX_TRESHOLD, 
            text_threshold=TEXT_TRESHOLD
        )

        annotated_frame = annotate(image_source=image, boxes=boxes, logits=logits, phrases=phrases)
        annotated_frame = annotated_frame[...,::-1] # BGR to RGB
        image_mask, box_list = generate_masks_with_grounding(image, boxes)
        ## Get the detection boxes
        ## For simplicity, here we are only using the first box, where ideally you can iter through all the boxes detected
        dino_box = box_list[0]
    except IndexError:
        print("No object found" + '.'*20)
    
    print("=================Dino detect done, segment start=================")
    mask_path = input_data['output_mask_image_dir']

    ## Inference
    predictor = SamPredictor(model['sam'])
    image_pil_arry = np.array(Image.open(io.BytesIO(img_bytes)).convert('RGB'))
    predictor.set_image(image_pil_arry)
    masks, _, _ = predictor.predict(
        box=dino_box[None, :],
        multimask_output=False,
    )
    ## save the mask
    up_img = Image.fromarray(np.invert(masks)[0])
    byteImgIO = io.BytesIO()
    up_img.save(byteImgIO, "WEBP")
    byteImgIO.seek(0)
    byteImg = byteImgIO.read()
    s3_resource = boto3.resource('s3')
    dir_lst = input_data['output_mask_image_dir'].split('/')
    img_id = uuid.uuid4().hex
    s3_bucket = dir_lst[2]
    s3_object_key = '/'.join(dir_lst[3:]) + img_id + '.webp'
    s3_resource.Bucket(s3_bucket).put_object(Key=s3_object_key, Body=byteImg, ContentType='image/webp')
    mask_image_output = 's3://{}/{}'.format(s3_bucket, s3_object_key)
    print("=================segment done=================")
    return mask_image_output

    
def output_fn(prediction, content_type):
    print("=================output process start=================")
    res = {'result': prediction}

    print("=================output process done=================")
    return json.dumps(res)
