# Train/Fine Tune SAM 2 on LabPics 1 dataset
# This mode use several images in a single batch
# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

import numpy as np
import torch
import cv2
import os
import LabPics1Reader
import LabPics2Reader
import Trans10KReader
import CocoVesselReader
from torch.onnx.symbolic_opset11 import hstack
import argparse
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
# Read data





parser = argparse.ArgumentParser(description='Evaluate The trained net on the LabPics benchmark')
parser.add_argument('--labpics_test_dir',default=r"/media/deadcrow/6TB/Data_zoo/LabPics_2/LabPics Chemistry//Test//",type=str,help="input dir")
parser.add_argument('--model_path',default=r"logs/model_small.torch",type=str,help="path to train model")
parser.add_argument('--sam2_checkpoint',default=r"sam2_hiera_small.pt",type=str,help="path to sam2 model weight")
parser.add_argument('--model_cfg',default=r"sam2_hiera_s.yaml",type=str,help=" model config")
parser.add_argument('--mode',default=r"labpics2",type=str,help=" could be labpics1 or labpics2")
args = parser.parse_args()

args = parser.parse_args()


# labpics1_dir=r"/media/deadcrow/6TB/Data_zoo/LabPicsV1//Simple/Test/" # Path to dataset (LabPics 1)
# labpics2_chemistry_dir=r"/media/deadcrow/6TB/Data_zoo/LabPics_2/LabPics Chemistry//Test//" # Path to dataset (LabPics 1)
# labpics2_medical_dir=r"/media/deadcrow/6TB/Data_zoo/LabPics_2/LabPics Medical//Test//" # Path to dataset (LabPics 1)
# mode="labpics1"
# sam2_checkpoint = "sam2_hiera_small.pt" # path to model weight
# model_cfg = "sam2_hiera_s.yaml" #  model config
#mode="labpics2"



#reader = LabPics1Reader.reader(labpics1_dir,train_mode=False)
if args.mode=="labpics2":
    reader = LabPics2Reader.reader(args.labpics_test_dir,train_mode=False)
elif args.mode== "labpics1":
    reader = LabPics1Reader.reader(args.labpics_test_dir, train_mode=False)
else:
    print("mode must be labpics2 or labpics1 ")
    exit()


# Load model


sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device="cuda") # load model
predictor = SAM2ImagePredictor(sam2_model)
predictor.model.load_state_dict(torch.load(args.model_path))
# Set training parameters

# predictor.model.sam_mask_decoder.train(False) # enable training of mask decoder
# predictor.model.sam_prompt_encoder.train(False) # enable training of prompt encoder
# predictor.model.image_encoder.train(False) # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
# optimizer = torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
# scaler = torch.cuda.amp.GradScaler() # mixed precision
iou_by_class = {"vessel":[],"filled":[],"transparent":[]}
# Training loop
for itr in range(400000):
    print(itr)
    with torch.no_grad(): # cast to mix precision
            if reader.epoch>1: break
            image, mask, ROI, input_label = reader.read_batch_vessel(batch_size=1)
            if mask.shape[0]==0: continue # ignore empty batches
            predictor.set_image_batch(image) # apply SAM image encoder to the image

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(np.ones([input_label.shape[0],1,2]), input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            # mask decoder

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"],image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=False,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            ROI = torch.tensor(ROI.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks)# Turn logit map to probability map
          ###  seg_loss = (ROI* (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001))).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            gt_mask2 =  gt_mask * ROI
            prd_mask2 = prd_mask * ROI
            inter = (gt_mask2 * (prd_mask2 > 0.5)).sum(2).sum(2)
            sm=(gt_mask2.sum(2).sum(2) + (prd_mask2 > 0.5).sum(2).sum(2) - inter)
            score_loss =0
            iou = inter / (sm + 0.000001)
            print(itr, ") Mean IOU by class:")
            for ii in range(sm.shape[1]):
                  if sm[:,ii].sum()>0:
                            score_loss += torch.abs(prd_scores[:,ii] - iou[:,ii]).mean()
                            iou_by_class[{0:"vessel",1:"filled",2:"transparent"}[ii]].append(np.mean(iou[:,ii].mean().cpu().detach().numpy()))
                            print("Mean IOU ", {0:"vessel",1:"filled",2:"transparent"}[ii],np.mean(iou_by_class[{0:"vessel",1:"filled",2:"transparent"}[ii]]))

            # Display results

            if itr==0: mean_iou=0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

           # print("step)",itr, "Accuracy(IOU)=",mean_iou)
           # print(iou_by_class)
