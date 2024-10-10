import cv2
import numpy as np
#############################################################################################################################
# Crop and resize image and mask and Object mask to feet batch size
def CropResize2(Img, MatMasks, ROImask, Hb, Wb):
    # ========================resize image if it too small to the batch size==================================================================================
    bbox = cv2.boundingRect(ROImask.astype(np.uint8))
    [h, w, d] = Img.shape
    Rs = np.max((Hb / h, Wb / w))
    Wbox = int(np.floor(bbox[2]))  # ROI Bounding box width
    Hbox = int(np.floor(bbox[3]))  # ROI Bounding box height
    if Wbox == 0: Wbox += 1
    if Hbox == 0: Hbox += 1

    Bs = np.min((Hb / Hbox, Wb / Wbox))
    if Rs > 1 or (
            Bs < 1 and np.random.rand() < 0.3):  # or Bs < 1 or np.random.rand() < 0.3:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
        h = int(np.max((h * Rs, Hb)))
        w = int(np.max((w * Rs, Wb)))
        Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

        MatMasks = cv2.resize(MatMasks.astype(float), dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        ROImask = cv2.resize(ROImask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
        bbox = (np.float32(bbox) * Rs.astype(np.float32)).astype(np.int32)

    # =======================Crop image to fit batch size===================================================================================
    x1 = int(np.floor(bbox[0]))  # Bounding box x position
    Wbox = int(np.floor(bbox[2]))  # Bounding box width
    y1 = int(np.floor(bbox[1]))  # Bounding box y position
    Hbox = int(np.floor(bbox[3]))  # Bounding box height

    if Wb > Wbox:
        Xmax = np.min((w - Wb, x1))
        Xmin = np.max((0, x1 - (Wb - Wbox) - 1))
    else:
        Xmin = x1
        Xmax = np.min((w - Wb, x1 + (Wbox - Wb) + 1))

    if Hb > Hbox:
        Ymax = np.min((h - Hb, y1))
        Ymin = np.max((0, y1 - (Hb - Hbox) - 1))
    else:
        Ymin = y1
        Ymax = np.min((h - Hb, y1 + (Hbox - Hb) + 1))

    if Ymax <= Ymin:
        y0 = Ymin
    else:
        y0 = np.random.randint(low=Ymin, high=Ymax + 1)

    if Xmax <= Xmin:
        x0 = Xmin
    else:
        x0 = np.random.randint(low=Xmin, high=Xmax + 1)

    # Img[:,:,1]*=PartMask
    # misc.imshow(Img)

    Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
    MatMasks = MatMasks[y0:y0 + Hb, x0:x0 + Wb]
    ROImask = ROImask[y0:y0 + Hb, x0:x0 + Wb]
    # ------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
    if not (Img.shape[0] == Hb and Img.shape[1] == Wb):
        Img = cv2.resize(Img, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)
        MatMasks = cv2.resize(MatMasks[y0:y0 + Hb, x0:x0 + Wb, :], interpolation=cv2.INTER_NEAREST)
        ROImask = cv2.resize(ROImask, dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    return Img, MatMasks, ROImask
    # misc.imshow(Img)


#################################################Generate Annotaton mask#############################################################################################################333


######################################################Augmented mask##################################################################################################################################
# Augment image
def Augment2(Img, MatMasks, ROIMask, prob=1):
    Img = Img.astype(np.float32)
    if np.random.rand() < 0.5:  # flip left right
        Img = np.fliplr(Img)
        ROIMask = np.fliplr(ROIMask)
        MatMasks = np.fliplr(MatMasks)

    if np.random.rand() < 0.0:  # flip up down
        Img = np.flipud(Img)
        ROIMask = np.flipud(ROIMask)
        MatMasks = np.flipud(MatMasks)
    #
    # if np.random.rand() < prob: # resize
    #     r=r2=(0.6 + np.random.rand() * 0.8)
    #     if np.random.rand() < prob*0.2:  #Strech
    #         r2=(0.65 + np.random.rand() * 0.7)
    #     h = int(PartMask.shape[0] * r)
    #     w = int(PartMask.shape[1] * r2)
    #     Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    #     PartMask = cv2.resize(PartMask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    #     AnnMap = cv2.resize(AnnMap.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
    if np.random.rand() < 0.035:  # Add noise
        noise = np.random.rand(Img.shape[0], Img.shape[1], Img.shape[2]) * 0.2 + np.ones(Img.shape) * 0.9

        Img *= noise
        Img[Img > 255] = 255
        #
    if np.random.rand() < 0.2:  # Gaussian blur
        Img = cv2.GaussianBlur(Img, (5, 5), 0)

    if np.random.rand() < 0.25:  # Dark light
        Img = Img * (0.5 + np.random.rand() * 0.65)
        Img[Img > 255] = 255
    # if np.random.rand() < prob:  # Dark light
    #     Img = Img * (0.5 + np.random.rand() * 0.7)
    #     Img[Img>255]=255

    if np.random.rand() < 0.2:  # GreyScale
        Gr = Img.mean(axis=2)
        r = np.random.rand()

        Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
        Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
        Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)

    return Img, MatMasks, ROIMask

#######################################################################################################################################################
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size
def CropResize(Img, AnnMap,Hb,Wb):
    # ========================resize image if it too small to the batch size==================================================================================

    h,w,d=Img.shape
    Bs = np.min((h/Hb,w/Wb))
    if Bs<1 or Bs>1.5:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
        h = int(h / Bs)+1
        w = int(w / Bs)+1
        Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
        for ky in AnnMap:
             AnnMap[ky] = cv2.resize(AnnMap[ky], dsize=(w, h), interpolation=cv2.INTER_NEAREST)
# =======================Crop image to fit batch size===================================================================================


    if np.random.rand()<0.9:

        if w>Wb:
            X0 = np.random.randint(w-Wb)
        else:
            X0 = 0
        if h>Hb:
            Y0 = np.random.randint(h-Hb)
        else:
            Y0 = 0

        Img=Img[Y0:Y0+Hb,X0:X0+Wb,:]
        for ky in AnnMap:
               AnnMap[ky] = AnnMap[ky][Y0:Y0+Hb,X0:X0+Wb]


    if not (Img.shape[0]==Hb and Img.shape[1]==Wb):

        Img = cv2.resize(Img, dsize=(Wb, Hb), interpolation=cv2.INTER_LINEAR)
        for ky in AnnMap:
            AnnMap[ky] = cv2.resize(AnnMap[ky], dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    return Img,AnnMap
    # misc.imshow(Img)
######################################################Augmented Image for training##################################################################################################################################
def Augment(Img,AnnMask):
    Img=Img.astype(np.float32)
    if np.random.rand()<0.5: # flip left right
        Img=np.fliplr(Img)
        for ky in AnnMask:
           AnnMask[ky] = np.fliplr(AnnMask[ky])
    if np.random.rand()<0.5:
        Img = Img[..., :: -1]
    if np.random.rand()< 0.08: # flip up down
        Img=np.flipud(Img)
        for ky in AnnMask:
            AnnMask[ky] =np.flipud(AnnMask[ky])
    # if np.random.rand()< 0.08: # Rotate
    #     Img=np.rot90(Img)
    #     for ky in AnnMask:
    #         AnnMask[ky] = np.rot90(AnnMask[ky])




    # if np.random.rand() < 0.03: # resize
    #     r=r2=(0.3 + np.random.rand() * 1.7)
    #     if np.random.rand() < 0.1:
    #         r2=(0.5 + np.random.rand())
    #     h = int(Img.shape[0] * r)
    #     w = int(Img.shape[1] * r2)
    #     Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
    #     for ky in AnnMask:
    #         AnnMask[ky] = cv2.resize(AnnMask[ky], dsize=(w, h), interpolation=cv2.INTER_NEAREST)

    # if np.random.rand() < prob/3: # Add noise
    #     noise = np.random.rand(Img.shape[0],Img.shape[1],Img.shape[2])*0.2+np.ones(Img.shape)*0.9
    #     Img *=noise
    #     Img[Img>255]=255
    #
    if np.random.rand() < 0.1: # Gaussian blur
        Img = cv2.GaussianBlur(Img, (5, 5), 0)

    if np.random.rand() < 0.07:  # Dark light
        Img = Img * (0.5 + np.random.rand() * 0.65)
        Img[Img>255]=255

    if np.random.rand() < 0.10:  # GreyScale
        Gr=Img.mean(axis=2)
        r=np.random.rand()

        Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
        Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
        Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


    return Img,AnnMask
########################################################################################################################################################