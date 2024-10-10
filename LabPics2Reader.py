
import os
import cv2
import numpy as np
import Augment_Crop

class reader():
    def __init__(self,data_dir, classes = ["vessel","filled","transparent","ROI"],train_mode=True):
            self.classes = classes
            self.data=[] # list of files in dataset
            self.train_mode = train_mode
            self.itr = 0
            self.epoch = 1
            for ff, name in enumerate(os.listdir(data_dir)):  # go over all folder annotation
                self.data.append(
                    {"image":data_dir+"/"+name +"/Image.jpg",
                     "vessel": data_dir +"/"+ name + "/SemanticMaps/FullImage/Vessel.png" ,
                     "filled": data_dir + "/"+name + "/SemanticMaps/FullImage/Filled.png",
                     "transparent": data_dir +name +"/SemanticMaps/FullImage/Transparent.png" ,
                     "ROI": data_dir +"//"+name + "/Ignore.png"
                     })



#################################################################################################################################################

    def read_single_vessel(self): # read random image and single mask from  the dataset (LabPics)

   #  select image
       # print("itr",self.itr)
        if self.train_mode:
            ent = self.data[np.random.randint(len(self.data))]
        else:
            ent = self.data[self.itr]  # choose random entry
            self.itr += 1
            if self.itr >= len(self.data):
                self.itr = 0
                self.epoch += 1

        Img = cv2.imread(ent["image"])[...,::-1]  # read image
        masks={}
        for cl in self.classes:
            if os.path.exists(ent[cl]):
                masks[cl] = cv2.imread(ent[cl])[:,:,0]
            else:
                masks[cl] = np.zeros_like(Img[:,:,0],dtype=np.float32)
        masks["ROI"]=1-masks["ROI"]


        if self.train_mode:
                Wb = np.random.randint(300, 1024)
                Hb = int(Wb * (1 - np.random.rand() * 0.3))
                Img, masks  = Augment_Crop.CropResize(Img, masks, Hb, Wb)
                Img, masks  = Augment_Crop.Augment(Img, masks)
                Img = Img.astype(np.uint8)
        else:
            r=np.min([1024/Img.shape[0],1024/Img.shape[1]])
            if r<1:
                h = int(Img.shape[0]*r) - 1
                w = int(Img.shape[0]*r) - 1
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
        masks[:, 2, :, :] = lmasks["transparent"]
        ROI = np.zeros([batch_size,3, 1024, 1024], dtype=np.float32)
        for i in range(3):
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