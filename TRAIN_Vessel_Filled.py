# Train/Fine  SAM2 for veseels and their content:
# 1) vessel/container region
# 2) Filled regions inside the vessel
# 3) Transparent regions

# Labpics can be downloaded from: https://zenodo.org/records/3697452/files/LabPicsV1.zip?download=1

import numpy as np
import torch
import argparse
import os
import LabPics1Reader
import LabPics2Reader
import Trans10KReader
import CocoVesselReader
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


parser = argparse.ArgumentParser(description='Train SAM2 On sementic segmentation of containers and fill level')
parser.add_argument('--labpics1_dir',default=r"/media/deadcrow/6TB/Data_zoo/LabPicsV1//Simple/Train/",type=str,help="path to dataset dir for labpics1 simple train set")
parser.add_argument('--labpics2_chemistry_dir',default=r"/media/deadcrow/6TB/Data_zoo/LabPics_2/LabPics Chemistry/Train/",type=str,help="path to dataset dir for labpics2 chemistry train set")
parser.add_argument('--labpics2_medical_dir',default=r"/media/deadcrow/6TB/Data_zoo/LabPics_2/LabPics Medical/Train/",type=str,help="path to dataset dir for labpics2 medical train set")
parser.add_argument('--coco_vessel_dir',default=r"/media/deadcrow/6TB/Data_zoo/COCO_Vessels_SemanticMaps/",type=str,help="path subset of coco dataset containing only vessels optional leave empty if unused")
parser.add_argument('--trans10k_dir',default=r"/media/deadcrow/6TB/Data_zoo/Trans10k/train/",type=str,help="path to trans10k (optional leave empty if unused)")
parser.add_argument('--sam2_checkpoint',default=r"sam2_hiera_small.pt",type=str,help="path to sam2 model standart checkpoint")
parser.add_argument('--model_cfg',default=r"sam2_hiera_s.yaml",type=str,help=" model config")
parser.add_argument('--weight_decay', default= 4e-5, type=float, help='optimizer weight decay')
parser.add_argument('--learning_rate', default= 1e-5, type=float, help='optimizer learning rate')
parser.add_argument('--log_dir', default= r"logs/", type=str, help='log folder were train model will be saved')
parser.add_argument('--batch_size', default= 6, type=str, help='batch size')
#parser.add_argument('--auto_resume', default= True, type=bool, help='start training from existing last saved model (Defult.torch)')

args = parser.parse_args()


#
# sam2_checkpoint = "sam2_hiera_small.pt" # path to model weight
# model_cfg = "sam2_hiera_s.yaml" #  model config
# # # Read data
#
# labpics1_dir=r"/media/deadcrow/6TB/Data_zoo/LabPicsV1//Simple/Train/" # Path to dataset (LabPics 1)
# labpics2_chemistry_dir=r"/media/deadcrow/6TB/Data_zoo/LabPics_2/LabPics Chemistry/Train/" # Path to dataset (LabPics 2 chemistry)
# labpics2_medical_dir=r"/media/deadcrow/6TB/Data_zoo/LabPics_2/LabPics Medical/Train/" # Path to dataset (LabPics 2 medical)
# trans10k_dir  = r"/media/deadcrow/6TB/Data_zoo/Trans10k/train/" #path to trans10k
# coco_vessel_dir = r"/media/deadcrow/6TB/Data_zoo/COCO_Vessels_SemanticMaps/"
# log_dir="logs_Vessel_filled_only_large_with_trans10k/"
# batch_size=6
if not os.path.exists(args.log_dir):
        os.mkdir(args.log_dir)

if not args.labpics1_dir=="":
       labpics1reader=LabPics1Reader.reader(args.labpics1_dir)
if not args.labpics2_chemistry_dir=="":
       labpics2reader_chem = LabPics2Reader.reader(args.labpics2_chemistry_dir)
if not args.labpics2_medical_dir=="":
      labpics2reader_med = LabPics2Reader.reader(args.labpics2_medical_dir)
if not args.coco_vessel_dir=="":
        coco_reader  = CocoVesselReader.reader(args.coco_vessel_dir)
if not args.trans10k_dir=="":
        trans10kreader = Trans10KReader.reader(args.trans10k_dir)


# Load model
# sam2_checkpoint = "sam2_hiera_large.pt" # path to model weight
# model_cfg = "sam2_hiera_l.yaml" #  model config

sam2_model = build_sam2(args.model_cfg, args.sam2_checkpoint, device="cuda") # load model
predictor = SAM2ImagePredictor(sam2_model)

# Set training parameters

predictor.model.sam_mask_decoder.train(True) # enable training of mask decoder
predictor.model.sam_prompt_encoder.train(True) # enable training of prompt encoder
predictor.model.image_encoder.train(True) # enable training of image encoder: For this to work you need to scan the code for "no_grad" and remove them all
optimizer = torch.optim.AdamW(params=predictor.model.parameters(),lr=1e-5,weight_decay=4e-5)
scaler = torch.cuda.amp.GradScaler() # mixed precision
iou_by_class = {"vessel":0,"filled":0,"transparent":0}
# Training loop

for itr in range(400000):
    with torch.cuda.amp.autocast(): # cast to mix precision
            #-----Choose reader and read data
            p=np.random.rand()

            #image,mask,input_point, input_label = read_batch(data,batch_size=4) # load data batch
            if p < 0.4:
               image, mask, ROI, input_label = labpics1reader.read_batch_vessel( batch_size=args.batch_size)
            elif p <0.70:
                image, mask, ROI, input_label = labpics2reader_chem.read_batch_vessel(batch_size=args.batch_size)
            elif p < 0.88:
                  if args.coco_vessel_dir == "": continue
                  image, mask, ROI, input_label = labpics2reader_med.read_batch_vessel(batch_size=args.batch_size)
            elif p < 1:
                   image, mask, ROI, input_label = coco_reader.read_batch_vessel(batch_size=args.batch_size)
            # else:
            #          if args.trans10k_dir == "": continue
            #         image, mask, ROI, input_label = trans10kreader.read_batch_vessel(batch_size=batch_size)
            if mask.shape[0]==0: continue # ignore empty batches

            #---- Run the image encoder

            predictor.set_image_batch(image) # apply SAM image encoder to the image
            # predictor.get_image_embedding()
            # prompt encoding

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(np.ones([input_label.shape[0],1,2]), input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            # run mask decoder

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"],image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=False,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution

            # Segmentaion Loss caclulation

            gt_mask = torch.tensor(mask.astype(np.float32)).cuda()
            ROI = torch.tensor(ROI.astype(np.float32)).cuda()
            prd_mask = torch.sigmoid(prd_masks)# Turn logit map to probability map
            seg_loss = (ROI* (-gt_mask * torch.log(prd_mask + 0.00001) - (1 - gt_mask) * torch.log((1 - prd_mask) + 0.00001))).mean() # cross entropy loss

            # Score loss calculation (intersection over union) IOU
            gt_mask2 =  gt_mask * ROI
            prd_mask2 = prd_mask * ROI
            inter = (gt_mask2 * (prd_mask2 > 0.5)).sum(2).sum(2)
            sm=(gt_mask2.sum(2).sum(2) + (prd_mask2 > 0.5).sum(2).sum(2) - inter)
            score_loss =0
            iou = inter / (sm + 0.000001)
            for ii in range(sm.shape[1]):
                  if sm[:,ii].sum()>0:
                            score_loss += torch.abs(prd_scores[:,ii] - iou[:,ii]).mean()
                            iou_by_class[{0:"vessel",1:"filled",2:"transparent"}[ii]] = iou_by_class[{0:"vessel",1:"filled",2:"transparent"}[ii]] * 0.99 + 0.01 * np.mean(iou[:,ii].mean().cpu().detach().numpy())
            loss=seg_loss#+score_loss*0.05  # mix losses

            # apply back propogation

            predictor.model.zero_grad() # empty gradient
            scaler.scale(loss).backward()  # Backpropogate
            scaler.step(optimizer)
            scaler.update() # Mix precision

            if itr%1000==0:
                    torch.save(predictor.model.state_dict(), args.log_dir+"/Defult.torch") # save model temp
            if itr%20000==0:
                    torch.save(predictor.model.state_dict(), args.log_dir+"/"+str(itr)+"Defult.torch") # save model

            # Display loss

            if itr==0: mean_iou=0
            mean_iou = mean_iou * 0.99 + 0.01 * np.mean(iou.cpu().detach().numpy())

            print("step)",itr, "Accuracy(IOU)=",mean_iou)
            print(iou_by_class)