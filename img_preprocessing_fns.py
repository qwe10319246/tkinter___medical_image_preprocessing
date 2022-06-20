# -*- coding: utf-8 -*-


import cv2
import numpy as np
import os
import scipy.signal
from scipy import ndimage
from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from PIL import Image


def test_import():
    print("test")


# algorithm fns

#RGB轉灰階
def fns_RGB2Gray(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return 0.299 * R + 0.58 * G + 0.114 * B

#呼叫RGB轉灰階function
def fns_callRGB2Gray():
    img = fns_RGB2Gray()
    return img
    
#負片效果(灰階)    
# def fns_negative_gray():
#     img = fns_RGB2Gray()
#     img = np.array(img)
#     img = 255 - img
#     return img

#負片效果
def fns_negative(img):
    img = np.array(img)
    img = 255 - img
    return img

#直方圖
def fns_histogram_eq(img):
    R, G, B = cv2.split(img)
    output1_R = cv2.equalizeHist(R)
    output1_G = cv2.equalizeHist(G)
    output1_B = cv2.equalizeHist(B)
    img = cv2.merge((output1_R, output1_G, output1_B))
    
    return img

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def fns_find_gradient(img, hx, hy):
    
    img = np.uint8(img)
    if len(img.shape) == 3:
    # print(len(img.shape))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    pdx = scipy.signal.correlate2d(img, hx, mode="same", boundary="symm")
    pdy = scipy.signal.correlate2d(img, hy, mode="same", boundary="symm")
    # result_img = np.add(np.abs(pdx), np.abs(pdy))
    # print(result_img)
    return np.add(np.abs(pdx), np.abs(pdy))

#各種邊緣偵測方法

#Roberts算子
def robert_op(img):
    hx = np.array(
        [
            [1, 0],
            [0, -1],
        ]
    )
    hy = np.array(
        [
            [0, -1],
            [1, 0],
        ]
    )
    
    return fns_find_gradient(img, hx, hy)
    

#Prewitt算子
def prewitt_op(img):
    hx = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1],
        ]
    )
    hy = np.array(
        [
            [1, 0, -1],
            [1, 0, -1],
            [1, 0, -1],
        ]
    )
    return fns_find_gradient(img, hx, hy)

#Sobel算子
def sobel_op(img):
    hx = np.array(
        [
            [1, 2, 1],
            [0, 0, 0],
            [-1, -2, -1],
        ]
    )
    hy = np.array(
        [
            [1, 0, -1],
            [2, 0, -2],
            [1, 0, -1],
        ]
    )
    return fns_find_gradient(img, hx, hy)

#Laplacian算子
def fns_laplacian_op(img):
    
    img = np.uint8(img)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    kernel = [
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1],
    ]

    img = scipy.signal.correlate2d(img, kernel, mode="same", boundary="symm")
    return img

#Canny算子
def fns_canny(img):  # sourcery no-metrics
    img = np.uint8(img)
    img = cv2.Canny(img, 30, 150)
    # print(img.shape)
    return img

def fns_black_noise(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 3:
        rows, cols, chn = img.shape
        
        for i in range(2000):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            img[x, y, :] = 0
            
    elif len(img.shape) == 2:
        rows, cols = img.shape
        
        for i in range(2000):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            img[x, y] = 0
    
    return img

def fns_white_noise(img):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if len(img.shape) == 3:
        rows, cols, chn = img.shape
        
        for i in range(2000):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            img[x, y, :] = 255
            
    elif len(img.shape) == 2:
        rows, cols = img.shape
        
        for i in range(2000):
            x = np.random.randint(0, rows)
            y = np.random.randint(0, cols)
            img[x, y] = 255
    
    
    return img

def morphology_erode(img):
    img = img.astype('uint8')
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    kernel = np.ones((3,3), np.uint8)
    img = cv2.erode(img, kernel, iterations = 1)
    
    return img

def morphology_dilate(img):
    img = img.astype('uint8')
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    kernel = np.ones((3,3), np.uint8)
    img = cv2.dilate(img,kernel, iterations = 1)
    return img

def morphology_open(img):
    img = img.astype('uint8')
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return img

def morphology_close(img):
    img = img.astype('uint8')
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
    kernel = np.ones((3,3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    return img

def morphology_thinning(img):
    img = img.astype('uint8')
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Structuring Element
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    # Create an empty output image to hold values
    thin = np.zeros(img.shape,dtype='uint8')

    # Loop until erosion leads to an empty set
    while (cv2.countNonZero(img)!=0):
        # Erosion
        erode = cv2.erode(img,kernel)
        # Opening on eroded image
        opening = cv2.morphologyEx(erode,cv2.MORPH_OPEN,kernel)
        # Subtract these two
        subset = erode - opening
        # Union of all previous sets
        thin = cv2.bitwise_or(subset,thin)
        # Set the eroded image for next iteration
        img = erode.copy()
    return thin

# 單尺度SSR(Single Scale Retinex)
def ssr(img, sigma=30):
    temp = cv2.GaussianBlur(img, (0,0), sigma)
    gaussian = np.where(temp==0, 0.01, temp)
    img_ssr = np.log10(img+0.01) - np.log10(gaussian)
    
    return img_ssr


#CLAHE應用在LAB顏色通道L
def clahe_l_channel_method(img, limit=3, grid=(8,8)):
    # print(img.dtype)
    img = np.uint8(img)
    image_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    l, a, b = cv2.split(image_lab)
    cl = clahe.apply(l)
    clahe_img = cv2.merge((cl,a,b))
    img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)
    
    return img

#圖片Gamma亮度校正後，再CLAHE應用在LAB顏色L通道
def gamma_clahe_l_channel_method(img, gamma=1.8, limit=3, grid=(8,8)):
#     image_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_bgr = adjust_gamma(img, gamma)
    
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    l, a, b = cv2.split(image_lab)
    cl = clahe.apply(l)
    clahe_img = cv2.merge((cl,a,b))
    img = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)
    
    return img

#顏色標準化
def color_normalize_method(img):
    img = np.uint8(img)
    image_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    b, g, r = cv2.split(image_bgr)
    norm_b = cv2.normalize(b, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
    norm_g = cv2.normalize(g, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
    norm_r = cv2.normalize(r, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
    img = cv2.merge((norm_b,norm_g,norm_r))
    
    return img

#CLAHE應用在BGR所有顏色通道上
def clahe_bgr_method(img, limit=3, grid=(8,8)):
    img = np.uint8(img)
    image_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    b, g, r = cv2.split(image_bgr)
    cb = clahe.apply(b)
    cg = clahe.apply(g)
    cr = clahe.apply(r)
    
    img = cv2.merge((cb,cg,cr))
    
    return img

#CLAHE應用在RGB顏色G通道上
def clahe_bgr_g_channel_method(img, limit=3, grid=(8,8)):
    img = np.uint8(img)
    image_bgr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    b, g, r = cv2.split(image_bgr)
    cg = clahe.apply(g)
    
    img = cv2.merge((b,cg,r))
    
    return img

#CLAHE應用在HSV顏色V通道上
def clahe_hsv_v_channel_method(img, limit=3, grid=(8,8)):
    img = np.uint8(img)
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    h, s, v = cv2.split(image_hsv)
    cv = clahe.apply(v)
    
    img = cv2.merge((h,s,cv))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    
    return img

#Gamma亮度校正
def adjust_gamma(img, gamma=0.8):
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                for i in np.arange(0, 256)]).astype("uint8")
                
    # res_image = resize_image(img)
    # apply gamma correction using the lookup table
    return cv2.LUT(img, table)

# Kaggle APTOS 2019數據集處理的方法，關於判斷眼底照片亮度校正、裁剪眼球輪廓圖片的方法
def crop_image_from_gray(img,tol=7):
    img = np.uint8(img)
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol
        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0): # image is too dark so that we crop out everything,
            return img # return original image
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def circle_crop(img, sigmaX=30):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 


#高斯濾波
def gaussianblur(img, sigmaX=30):
    img = np.uint8(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

#轉換成opencv BGR色彩通道
def fns_RGB_to_BGR(img):
   img = np.uint8(img) 
   img =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
   return img 

IMAGE_SIZE = 200
def info_image(im):
    # Compute the center (cx, cy) and radius of the eye
    cy = im.shape[0]//2
    midline = im[cy,:]
    midline = np.where(midline>midline.mean()/3)[0]
    if len(midline)>im.shape[1]//2:
        x_start, x_end = np.min(midline), np.max(midline)
    else: # This actually rarely happens p~1/10000
        x_start, x_end = im.shape[1]//10, 9*im.shape[1]//10
    cx = (x_start + x_end)/2
    r = (x_end - x_start)/2
    return cx, cy, r

def resize_image(im, augmentation=False):
    # Crops, resizes and potentially augments the image to IMAGE_SIZE
    cx, cy, r = info_image(im)
    scaling = IMAGE_SIZE/(2*r)
    rotation = 0
    if augmentation:
        scaling *= 1 + 0.3 * (np.random.rand()-0.5)
        rotation = 360 * np.random.rand()
    M = cv2.getRotationMatrix2D((cx,cy), rotation, scaling)
    M[0,2] -= cx - IMAGE_SIZE/2
    M[1,2] -= cy - IMAGE_SIZE/2
    return cv2.warpAffine(im,M,(IMAGE_SIZE,IMAGE_SIZE)) # This is the most important line

def circle_crop2(img, sigmaX=30):   
    """
    Create circular crop around image centre    
    """    
    img = crop_image_from_gray(img)    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    height, width, depth = img.shape    
    
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    circle_img = np.zeros((height, width), np.uint8)
    cv2.circle(circle_img, (x,y), int(r), 1, thickness=-1)
    img = cv2.bitwise_and(img, img, mask=circle_img)
    img = crop_image_from_gray(img)
    # img=cv2.addWeighted(img,4, cv2.GaussianBlur( img , (0,0) , sigmaX) ,-4 ,128)
    return img 

def fns_medical_img_enhancement(img, sigmaX=30, gamma=0.8, limit=3, grid=(8,8)):
    img = np.uint8(img)
    b, g, r = cv2.split(img)
    
    # gamma hsv v channel
    image_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(image_hsv)
    image_gamma = adjust_gamma(image_hsv, gamma)
    gh, gs, gv = cv2.split(image_gamma)
    
    # gain matrix g
    # gamma v / original v
    gmatrix = np.divide(gv,v)
    gmatrix[np.isnan(gmatrix)] = 0
    
    # original bgr x gain_matrix_g
    gb = b * gmatrix
    gg = g * gmatrix
    gr = r * gmatrix
    lg_img = cv2.merge((gb,gg,gr))
    
    # float to int
    lg_img = np.uint8(lg_img)
    
    # clahe on lab l channel
    image_lab = cv2.cvtColor(lg_img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    l, a, b = cv2.split(image_lab)
    cl = clahe.apply(l)
    clahe_img = cv2.merge((cl,a,b))
    image_bgr = cv2.cvtColor(clahe_img, cv2.COLOR_LAB2BGR)

    # color normalize
    b, g, r = cv2.split(image_bgr)
    norm_b = cv2.normalize(b, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
    norm_g = cv2.normalize(g, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
    norm_r = cv2.normalize(r, None, alpha=0,beta=255, norm_type=cv2.NORM_MINMAX)
    
    img = cv2.merge((norm_b,norm_g,norm_r))
    
    return img
     

