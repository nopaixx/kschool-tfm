import os
import pylab as pl # linear algebra + plotting
import numpy as np

# trying to add random noise
from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random
import cv2
from skimage.transform import resize


from utils.first_model_globals import H,W,BATCH_SIZE,car_path, mask_path


#############FOÑDER TO DATA LOADER###################
def intel_files_names_in_path(path="",extension=".jpg"):
    files = []
    
    for f in os.listdir(path):
        filename, file_extension = os.path.splitext(f)
        if file_extension == '.jpg':
            files.append(path+filename+".jpg")
    
            
    return files


def files_names_in_path_carvana(path="",extension=".jpg"):
    files = []
    masks = []
    for f in os.listdir(path):
        filename, file_extension = os.path.splitext(f)
        if file_extension == '.gif':
            files.append(filename.replace('_mask','')+".jpg")
            masks.append(filename+".gif")
            
    return files, masks

##############################################################



################ AUGMENTATORS ###############################




import pylab as pl # linear algebra + plotting


# trying to add random noise
from skimage.transform import warp, AffineTransform, ProjectiveTransform
from skimage.exposure import equalize_adapthist, equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.filters import gaussian
from skimage.util import random_noise
import random



def downsample(img, h, w):
    ret = resize(img, (h, w), mode='constant', preserve_range=True)
    # plt.imshow(ret)
    return ret
    #return cv2.resize(img, (h, w))

def randRange(a, b):
    return pl.rand() * (b - a) + a

def randomPerspective(im,mask):

    region = 1/4
    A = pl.array([[0, 0], [0, im.shape[0]], [im.shape[1], im.shape[0]], [im.shape[1], 0]])
    B = pl.array([[int(randRange(0, im.shape[1] * region)), int(randRange(0, im.shape[0] * region))],
                  [int(randRange(0, im.shape[1] * region)), int(randRange(im.shape[0] * (1-region), im.shape[0]))],
                  [int(randRange(im.shape[1] * (1-region), im.shape[1])), int(randRange(im.shape[0] * (1-region), im.shape[0]))],
                  [int(randRange(im.shape[1] * (1-region), im.shape[1])), int(randRange(0, im.shape[0] * region))],
                 ])

    pt = ProjectiveTransform()
    pt.estimate(A, B)

    return warp(im, pt, output_shape=im.shape[:2]), warp(mask, pt, output_shape=im.shape[:2])


def augmentation(image, mascara):
    noise = np.random.randint(0,3)

    newimg = image
    newmask = mascara

    if noise == 1:
        #aplicamos randomroise
        var = np.random.randint(0,1000)/100000
        newimg = random_noise(image, mode='gaussian', var=var)
    elif noise == 2:
        #applicamos gaussian noise
        newimg = gaussian(image, sigma=randRange(0, 2))
    #sino no aplicamos ruidos dejamos la original

    crop = np.random.randint(0,2)

    if crop == 1:
        #aplicamos crop
        #print("CROP")
        #newimg, newmask = randomCrop(newimg, newmask)
    #elif crop == 2:
        #print("prespective")
        newimg, newmask = randomPerspective(newimg, newmask)

     ## ¿Nos falta algun agumentador que canvie el color?? bueno vamos a ver que tal va este
    return newimg, newmask


#######################################################


###########APPLY CAR TO BACKGROUND################


#Need only car image apply bitwise
def get_only_object(img, mask, back_img):
    fg = cv2.bitwise_or(img, img, mask=mask)
    #imshow(fg)
    # invert mask
    mask_inv = cv2.bitwise_not(mask)
    #fg_back = cv2.bitwise_or(back_img, back_img, mask=mask)
    fg_back_inv = cv2.bitwise_or(back_img, back_img, mask=mask_inv)
    #imshow(fg_back_inv)
    final = cv2.bitwise_or(fg, fg_back_inv)
    #imshow(final)

    return final

###################################################





