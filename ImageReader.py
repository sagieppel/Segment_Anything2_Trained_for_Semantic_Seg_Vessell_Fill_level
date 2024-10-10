
import os
import cv2
import numpy as np
import Augment_Crop

class reader():
    def __init__(self,data_dir):

            self.data=[] # list of files in dataset
            for ff, name in enumerate(os.listdir(data_dir)):  # go over all folder annotation
                self.data.append(
                    {"image":data_dir+name,
                     })
            self.itr=0

#################################################################################################################################################

    def read_single_vessel(self): # read random image and single mask from  the dataset (LabPics)

   #  select image


        ent  = self.data[self.itr] # choose random entry
        self.itr+=1
        print("itr",self.itr)
        Img = cv2.imread(ent["image"])[...,::-1]  # read image
        masks={}



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

    def read_batch_vessel(self, batch_size=1):
        limage = []
        lmasks = {}
        for i in range(batch_size):
            image, masks = self.read_single_vessel()
            limage.append(image)

        masks=np.zeros([batch_size,3,1024,1024],dtype=np.float32)
        ROI = np.zeros([batch_size,3, 1024, 1024], dtype=np.float32)

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