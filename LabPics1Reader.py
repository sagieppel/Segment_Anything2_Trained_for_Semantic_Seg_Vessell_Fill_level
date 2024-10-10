
import os
import cv2
import numpy as np
import Augment_Crop

class reader():
    def __init__(self,data_dir, classes = ["vessel","filled"],train_mode=True):
            self.train_mode=train_mode
            self.classes = classes
            self.epoch = 1
            self.data=[] # list of files in dataset
            for ff, name in enumerate(os.listdir(data_dir+"/Image/")):  # go over all folder annotation
                self.data.append(
                    {"image":data_dir+"/Image/"+name,
                     "annotation":data_dir+"/Instance/"+name[:-4]+".png",
                     "vessel": data_dir + "/Semantic/1_Vessel/" + name[:-4] + ".png",
                     "filled": data_dir + "/Semantic/16_Filled/" + name[:-4] + ".png"
                     })
            self.itr=0
   #  def read_single_mat(self): # read random image and single mask from  the dataset (LabPics)
   #
   # #  select image
   #      if self.train_mode:
   #           ent = self.data[np.random.randint(len(self.data))]
   #      else:
   #           ent  = self.data[self.itr] # choose random entry
   #           self.itr+=1
   #      Img = cv2.imread(ent["image"])[...,::-1]  # read image
   #      ann_map = cv2.imread(ent["annotation"]) # read annotation
   #
   # # # resize image
   # #
   # #      r = np.min([1024 / Img.shape[1], 1024 / Img.shape[0]]) # scalling factor
   # #      Img = cv2.resize(Img, (int(Img.shape[1] * r), int(Img.shape[0] * r)))
   # #      ann_map = cv2.resize(ann_map, (int(ann_map.shape[1] * r), int(ann_map.shape[0] * r)),interpolation=cv2.INTER_NEAREST)
   #
   #
   #
   # # merge vessels and materials annotations
   #
   #      mat_map = ann_map[:,:,0] # material annotation map
   #      ves_map = ann_map[:,:,2] # vessel  annotaion map
   #      mat_map[mat_map==0] = ves_map[mat_map==0]*(mat_map.max()+1) # merge maps
   #
   # # Get binary masks and points
   #
   #
   #      inds = np.unique(mat_map)[1:] # load all indices
   #      if inds.__len__()>0:
   #            ind = inds[np.random.randint(inds.__len__())]  # pick single segment
   #      else:
   #            return self.read_single()
   #            # for ind in inds:
   #      mask = (mat_map == ind).astype(np.uint8)  # make binary mask corresponding to index ind
   #
   #      if self.train_mode:
   #          Wb = np.random.randint(320, 1024)
   #          Hb = int(Wb * (1 - np.random.rand() * 0.3))
   #          mat_map[mat_map > 240] = 0
   #          Img, mask, z = Augment_Crop.CropResize2(Img, mask, mask, Hb, Wb)
   #          Img, mask, z = Augment_Crop.Augment2(Img, mask,mask)
   #          Img = Img.astype(np.uint8)
   #      else:
   #          r=np.min([1024/Img.shape[0],1024/Img.shape[1]])
   #          if r<1:
   #              h = int(Img.shape[0]*r) - 1
   #              w = int(Img.shape[0]*r) - 1
   #              Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
   #              for ky in mask:
   #                  masks[ky] = cv2.resize(masks[ky], dsize=(w, h), interpolation=cv2.INTER_NEAREST)
   #      #***************
   #      # cv2.imshow("image",Img)
   #      # Img[:,:,0][mask>0]=255
   #      # Img[:,:,1][mask>0]=0
   #      # cv2.imshow("marked",Img)
   #      # cv2.imshow("mask",mask*255)
   #      # cv2.waitKey()
   #      # #***************
   #
   #
   #
   #
   #      if Img.shape[0] < 1024:
   #          Img = np.concatenate([Img, np.zeros([1024 - Img.shape[0], Img.shape[1], 3], dtype=np.uint8)], axis=0)
   #          mask = np.concatenate([mask, np.zeros([1024 - mask.shape[0], mask.shape[1]], dtype=np.uint8)],axis=0)
   #      if Img.shape[1] < 1024:
   #          Img = np.concatenate([Img, np.zeros([Img.shape[0], 1024 - Img.shape[1], 3], dtype=np.uint8)], axis=1)
   #          mask = np.concatenate([mask, np.zeros([mask.shape[0], 1024 - mask.shape[1]], dtype=np.uint8)],axis=1)
   #
   #
   #
   #      coords = np.argwhere(mask > 0) # get all coordinates in mask
   #      yx = np.array(coords[np.random.randint(len(coords))]) # choose random point/coordinate
   #      return Img,mask,[[yx[1], yx[0]]]

########################################################################################################################################################

    # def read_batch_mat(self, batch_size=4):
    #     limage = []
    #     lmask = []
    #     linput_point = []
    #     for i in range(batch_size):
    #         image, mask, input_point = self.read_single()
    #         limage.append(image)
    #         lmask.append(mask)
    #         linput_point.append(input_point)
    #
    #     return limage, np.array(lmask), np.array(linput_point), np.ones([batch_size, 1])

#################################################################################################################################################

    def read_single_vessel(self): # read random image and single mask from  the dataset (LabPics)

   #  select image


        if self.train_mode:
                ent = self.data[np.random.randint(len(self.data))]
        else:
               ent  = self.data[self.itr] # choose random entry
               self.itr+=1
               if self.itr >= len(self.data):
                   self.itr = 0
                   self.epoch +=1
               print("itr",self.itr)
        Img = cv2.imread(ent["image"])[...,::-1]  # read image
        masks={}
        for cl in self.classes:
            if os.path.exists(ent[cl]):
                masks[cl] = cv2.imread(ent[cl],0)
                masks["ROI"] = (masks[cl]<3).astype(np.uint8)
                masks[cl][masks[cl] > 2] = 0
                masks[cl][masks[cl] > 0] = 1
            else:
                masks[cl] = np.zeros_like(Img[:,:,0],dtype=np.float32)
        if not "ROI" in masks:
                     return self.read_single_vessel()

        if self. train_mode:
            Wb = np.random.randint(300, 1024)
            Hb = int(Wb * (1 - np.random.rand() * 0.3))
            Img, masks  = Augment_Crop.CropResize(Img, masks, Hb, Wb)
            Img, masks  = Augment_Crop.Augment(Img, masks)
            Img = Img.astype(np.uint8)
        else:
         r = np.min([1024 / Img.shape[0], 1024 / Img.shape[1]])
         if r < 1:
            h = int(Img.shape[0] * r) - 1
            w = int(Img.shape[0] * r) - 1
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            for ky in masks:
                masks[ky] = cv2.resize(masks[ky], dsize=(w, h), interpolation=cv2.INTER_NEAREST)



        if Img.shape[0] < 1024:
            Img = np.concatenate([Img, np.zeros([1024 - Img.shape[0], Img.shape[1], 3], dtype=np.uint8)], axis=0)
            for cl in masks:
                masks[cl] = np.concatenate([masks[cl], np.zeros([1024 - masks[cl].shape[0], masks[cl].shape[1]], dtype=np.uint8)],axis=0)

        if Img.shape[1] < 1024:
            Img = np.concatenate([Img, np.zeros([Img.shape[0], 1024 - Img.shape[1], 3], dtype=np.uint8)], axis=1)
            for cl in masks:
                masks[cl] = np.concatenate([masks[cl], np.zeros([masks[cl].shape[0], 1024 - masks[cl].shape[1]], dtype=np.uint8)],axis=1)




        # ***************
        # cv2.imshow("image", Img)
        # for ky in masks:
        #         I = Img.copy()
        #
        #         I[:, :, 0][masks[ky] > 0] = 255
        #         I[:, :, 1][masks[ky] > 0] = 0
        #         cv2.imshow(ky, I)
        #         #cv2.imshow(ky+"mask",masks[ky]*255)
        # cv2.waitKey()
        # #***************



        return Img,masks

########################################################################################################################################################

    def read_batch_vessel(self, batch_size=4):
        limage = []
        lmasks = {}
        for i in range(batch_size):
            image, masks = self.read_single_vessel()
            for ky in masks:
                if not ky in lmasks:
                    lmasks[ky] = []
                lmasks[ky].append(masks[ky])
            limage.append(image)

        masks=np.zeros([batch_size,3,1024,1024],dtype=np.float32)
        masks[:, 0, :, :] = lmasks["vessel"]
        masks[:, 1, :, :] = lmasks["filled"]
        ROI = np.zeros([batch_size,3, 1024, 1024], dtype=np.float32)
        for i in range(2):
            ROI[:,i]=lmasks["ROI"]

#==========================================================================
        # for i in range(ROI.shape[0]):
        #     for ii in range(3):
        #         cv2.destroyAllWindows()
        #         cv2.imshow(str(i),limage[i])
        #         cv2.imshow("ROI" + str(i) + "_"+str(ii), ROI[i][ii]*255)
        #         cv2.imshow("mask" +  str(i) + "_"+str(ii), masks[i][ii]*255)
        #         cv2.waitKey()
#==============================================================================================


        return limage,  masks,ROI, np.ones([batch_size, 1])