# Run trained net on folder of images and display/save prediction

import numpy as np
import torch
import cv2
import os
import ImageReader
from torch.onnx.symbolic_opset11 import hstack

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
# Read data

import argparse
parser = argparse.ArgumentParser(description='Run SAM2 optimize for liquid level and vessls detection save and display results')
parser.add_argument('--in_dir',default=r"/media/deadcrow/6TB/Data_zoo/LabPicsV1/Simple/Test/Image/",type=str,help="input dir")
parser.add_argument('--model_path',default=r"logs/model_small.torch",type=str,help="path to train model")
parser.add_argument('--sam2_checkpoint',default=r"sam2_hiera_small.pt",type=str,help="path to sam2 model checkpoint (standart from main SAM2 repository)")
parser.add_argument('--model_cfg',default=r"sam2_hiera_s.yaml",type=str,help=" model config")
parser.add_argument('--outdir',default=r"outdir/",type=str,help=" path where output dir where the annotations will be saved, the script will also display  result on screen")
args = parser.parse_args()

# in_dir=r"/media/deadcrow/6TB/Data_zoo/LabPicsV1/Simple/Test/Image/" # Path to dataset (LabPics 1)
# sam2_checkpoint = "sam2_hiera_small.pt" # path to model weight
# model_cfg = "sam2_hiera_s.yaml" #  model config
# "logs_Vessel_only_small_model/100000Defult.torch"
# "display_dir"

reader = ImageReader.reader(args.in_dir)
#reader = LabPics2Reader.reader(labpics2_chemistry_dir,train_mode=False)
# kp2reader_med = LabPics2Reader.reader(labpics2_medical_dir)
# tans10kreader = Trans10KReader.reader(trans10k_dir)
# coco_reader  = CocoVesselReader.reader(coco_vessel_dir)


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
            p=np.random.rand()
            #image,mask,input_point, input_label = read_batch(data,batch_size=4) # load data batch
            #if p < 0.4:
            image, mask, ROI, input_label = reader.read_batch_vessel()
            # elif p <0.75:
            #     image, mask, ROI, input_label = kp2reader_chem.read_batch_vessel()
            # elif p < 0.88:
            #       image, mask, ROI, input_label = kp2reader_med.read_batch_vessel()
            # else:
            #        image, mask, ROI, input_label = coco_reader.read_batch_vessel()
            if mask.shape[0]==0: continue # ignore empty batches
            predictor.set_image_batch(image) # apply SAM image encoder to the image
            # predictor.get_image_embedding()
            # prompt encoding

            mask_input, unnorm_coords, labels, unnorm_box = predictor._prep_prompts(np.ones([input_label.shape[0],1,2]), input_label, box=None, mask_logits=None, normalize_coords=True)
            sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(points=(unnorm_coords, labels),boxes=None,masks=None,)

            # mask decoder

            high_res_features = [feat_level[-1].unsqueeze(0) for feat_level in predictor._features["high_res_feats"]]
            low_res_masks, prd_scores, _, _ = predictor.model.sam_mask_decoder(image_embeddings=predictor._features["image_embed"],image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),sparse_prompt_embeddings=sparse_embeddings,dense_prompt_embeddings=dense_embeddings,multimask_output=True,repeat_image=False,high_res_features=high_res_features,)
            prd_masks = predictor._transforms.postprocess_masks(low_res_masks, predictor._orig_hw[-1])# Upscale the masks to the original image resolution



            prd_mask = (torch.sigmoid(prd_masks).cpu().detach().numpy()>0.5)[0]# Turn logit map to probability map
            cv2.destroyAllWindows()
            if not os.path.exists(args.outdir):
                  os.mkdir(args.outdir)
            cv2.imwrite(args.outdir+"/"+str(itr)+".jpg",image[0][:, :, ::-1])
            for dd in range(3):
                im = image[0][:, :, ::-1].copy()
                im[:,:,0][prd_mask[dd]]=0
                im[:, :, 1][prd_mask[dd]] = 0
                cv2.imwrite(args.outdir + "/" + str(itr) + {0:"vessel",1:"filled",2:"transparent"}[dd]+".jpg", im)

                cv2.imshow({0:"vessel",1:"filled",2:"transparent"}[dd],im)
            cv2.waitKey()
