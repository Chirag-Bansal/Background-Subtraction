""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np

start_frame = 0
end_frame = 0

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args

def HistogramEqualization(img):
    hist,bins = np.histogram(img.flatten(),256,[0,256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    img2 = cdf[img]
    return img2

def MedianBlur(img):
    return cv2.medianBlur(img,5)

def BilateralFiltering(img):
    blur = cv2.bilateralFilter(img,9,75,75)
    return blur

def GaussianRemoveNoise(img, kernel):
    blur = cv2.GaussianBlur(img,(kernel,kernel),0)
    thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)[1]
    return thresh

def eval_frames(args):
    global end_frame
    global start_frame
    file = open(args.eval_frames, "r")
    numbers = file.readline()
    nums = numbers.split(' ')
    start_frame = int(nums[0])
    end_frame = int(nums[1])
    file.close()

def BackgroundMOG(args, equal):
    files_list = os.listdir(args.inp_path)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    # fgbg.setDetectShadows(False)
    eval_frames(args)
    for image in files_list:
        img = cv2.imread(os.path.join(args.inp_path, image))
        img_num = int(image[2:8])
        if(equal):
            img = HistogramEqualization(img)
        fgmask = fgbg.apply(img)
        thresh = GaussianRemoveNoise(fgmask,17)

        ret, binary_map = cv2.threshold(thresh,127,255,0)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
        areas = stats[1:,cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= 100:   #keep
                result[labels == i + 1] = 255

        image = "gt" + image[2:8] + ".png"
        if img_num <= end_frame and img_num >= start_frame:
            cv2.imwrite(os.path.join(args.out_path, image), result)

def BackgroundKNN(args, equal):
    files_list = os.listdir(args.inp_path)
    fgbg = cv2.createBackgroundSubtractorKNN()
    eval_frames(args)
    for image in files_list:
        img = cv2.imread(os.path.join(args.inp_path, image))
        img_num = int(image[2:8])
        if(equal):
            img = HistogramEqualization(img)

        fgmask = fgbg.apply(img)

        image = "gt" + image[2:8] + ".png"

        # thresh = BilateralFiltering(fgmask)
        # thresh = MedianBlur(fgmask)
        thresh = GaussianRemoveNoise(fgmask,19)

        # 0 0 0 1 0 
        # 0 0 0 0 0
        # 1 1 0 0 0 
        # 1 1 0 0 0
        # 0 1 0 0 1
        # 0 0 0 0 1

        ret, binary_map = cv2.threshold(thresh,127,255,0)
        nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
        areas = stats[1:,cv2.CC_STAT_AREA]
        result = np.zeros((labels.shape), np.uint8)
        for i in range(0, nlabels - 1):
            if areas[i] >= 100:   #keep
                result[labels == i + 1] = 255
        
        if img_num <= end_frame and img_num >= start_frame:
            cv2.imwrite(os.path.join(args.out_path, image), result)

def RunningAverage(args):
    eval_frames(args)
    files_list = os.listdir(args.inp_path)

    # original kernel -> kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel = np.ones((5,5), np.uint8)
    image1 = cv2.imread(os.path.join(args.inp_path, files_list[0]))
    average = np.float32(image1)
    counter = 0

    for image in files_list:
    
        if counter != 0:
            img = cv2.imread(os.path.join(args.inp_path, image))
            num = int(image[2:8])

            cv2.accumulateWeighted(img, average, 0.02)
            rf = cv2.convertScaleAbs(average) # background subtraction
            foreImg = cv2.absdiff(rf, img)
            fgmask = cv2.cvtColor(foreImg, cv2.COLOR_BGR2GRAY)
            ret, fgmask = cv2.threshold(fgmask, 45, 255, cv2.THRESH_BINARY)
            # original noise removal -> fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
            fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel)

            image = "gt" + image[2:8] + ".png"

            ret, binary_map = cv2.threshold(fgmask,127,255,0)
            nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, None, None, None, 8, cv2.CV_32S)
            areas = stats[1:,cv2.CC_STAT_AREA]
            result = np.zeros((labels.shape), np.uint8)
            for i in range(0, nlabels - 1):
                if areas[i] >= 100:   #keep
                    result[labels == i + 1] = 255

            if num <= end_frame and num >= start_frame:
                cv2.imwrite(os.path.join(args.out_path, image), result)
        else:
            counter += 1
            continue

def baseline_bgs(args):
    BackgroundKNN(args,False)

def illumination_bgs(args):
    BackgroundMOG(args,True)

def jitter_bgs(args):
    BackgroundMOG(args,False)

def dynamic_bgs(args):
    BackgroundMOG(args,False)

def ptz_bgs(args):
    BackgroundMOG(args,False)

def main(args):
    if args.category not in "bijmp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)