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

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
# Read data

labpics1_dir=r"/media/deadcrow/6TB/Data_zoo/LabPicsV1//Simple/Test/" # Path to dataset (LabPics 1)
labpics2_chemistry_dir=r"/media/deadcrow/6TB/Data_zoo/LabPics_2/LabPics Chemistry//Test//" # Path to dataset (LabPics 1)
labpics2_medical_dir=r"/media/deadcrow/6TB/Data_zoo/LabPics_2/LabPics Medical//Test//" # Path to dataset (LabPics 1)
mode="labpics1"
#mode="labpics2"



#reader = LabPics1Reader.reader(labpics1_dir,train_mode=False)
reader = LabPics2Reader.reader(labpics2_chemistry_dir,train_mode=False)
#reader_med = LabPics2Reader.reader(labpics2_medical_dir)




# Load model
model_dir = "logs_Vessel_only_large/"#/logs_Vessel_only_small_model/"
sam2_checkpoint = "sam2_hiera_large.pt"  # path to model weight
model_cfg = "sam2_hiera_l.yaml"  # model config
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device="cuda")  # load model
predictor = SAM2ImagePredictor(sam2_model)
if os.path.exists(model_dir + "/results.txt"): exit("results file exists")
with open(model_dir+"/results.txt", 'w') as fl: fl.write(model_dir)
fl.close()

for fl in os.listdir(model_dir):
    if not ".torch" in fl: continue
    predictor.model.load_state_dict(torch.load(model_dir+"/"+fl))
    iou_by_class = {"vessel":[],"filled":[],"transparent":[]}
    # Training loop
    reader.epoch = 1
    reader.itr = 0
    while (reader.epoch==1):
        print(reader.itr)
        with torch.no_grad(): # cast to mix precision
                image, mask, ROI, input_label = reader.read_batch_vessel(batch_size=3)
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
                txt="\n\n"
                txt+=str(reader.itr)+"\t"+fl+ "\tMean IOU by class:\n"
              #  print(reader.itr,")",fl, ") Mean IOU by class:")
                for ii in range(sm.shape[1]):
                      if sm[:,ii].sum()>0:
                                score_loss += torch.abs(prd_scores[:,ii] - iou[:,ii]).mean()

                                iou_by_class[{0:"vessel",1:"filled",2:"transparent"}[ii]].append(np.mean(iou[:,ii].mean().cpu().detach().numpy()))
                                txt+={0:"vessel",1:"filled",2:"transparent"}[ii]+"\t"+str(np.mean(iou_by_class[{0:"vessel",1:"filled",2:"transparent"}[ii]])) +"\n"
                              #  print("Mean IOU ", {0:"vessel",1:"filled",2:"transparent"}[ii],np.mean(iou_by_class[{0:"vessel",1:"filled",2:"transparent"}[ii]]))
                print(txt)

    with open(model_dir+"/results.txt", 'a') as fl: fl.write(txt)
    fl.close()
