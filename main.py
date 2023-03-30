## import libs
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from ultralytics import YOLO
import cv2

import glob
import os


## download a weigth
path_to_model = 'yolov8_segment_572images_custom_pothole_bs=16_best.pt'
model = YOLO(path_to_model)


# meanshift
def meanshif(data):
    bandwidth = estimate_bandwidth(data, quantile=0.3, n_samples=300)
    ms = MeanShift(bandwidth=bandwidth)
    image = ms.fit(data)
    print('gogo')
    cv2.imwrite(f"111", image)

# the file name of the dataset



## use model
def predict_image(path_to_image):
    result = model.predict(source=path_to_image, conf=0.05, save=True)

    img = result[0].plot()
    b_mask = []

    ##use with mask segmentation
    # for i in range(15):
    #     try:
    #         print("add segment")
    #         b_mask.append((result[0].masks.masks[i].numpy() * 255).astype("uint8"))
    #     except:
    #         print("Index out of range")
    #         break

    boxes = result[0]
    return img, boxes, b_mask


def delete_the_background(mask, image, filename):
    for index, segment in enumerate(mask):

        # img = cv2.imread(image)
        img = image

        # compute the bitwise AND using the mask
        masked_img = cv2.bitwise_and(img, img, mask=segment)

        # Convert image to image gray
        tmp = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

        # Applying thresholding technique
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)

        # Using cv2.split() to split channels
        # of coloured image
        b, g, r = cv2.split(masked_img)

        # Making list of Red, Green, Blue
        # Channels and alpha
        rgba = [b, g, r, alpha]

        # Using cv2.merge() to merge rgba
        # into a coloured/multi-channeled image
        dst = cv2.merge(rgba, 4)

        # Writing and saving to a new image
        cv2.imwrite(f"{filename}{index}.png", dst)


def resize_image(image):
    # img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img = cv2.imread(image)
    print('Original Dimensions : ', img.shape)

    width = int(img.shape[1] / (img.shape[1]/640))
    height = int(img.shape[0] / (img.shape[1]/640))
    # dim = (width, height)
    dim = (1280, 960)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    # resized = cv2.resize(image, dim)

    cv2.imwrite(f"{image}", resized)

    print('Resized Dimensions : ', resized.shape)


if __name__ == '__main__':
    # image_path = 'images/3_2023-02-28__12-02-39-667_IMX298_ColorCalib_2023-02-28__12-02-39-762_exp-6.0_gain1.0_f144.0.png'
    # result_img, bboxes_string, mask = predict_image(image_path)
    # cv2.imshow("Ship_detection", result_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    src_dir = "3d_image"
    for jpgfile in glob.iglob(os.path.join(src_dir, "*.png")):
        # resize_image(jpgfile)
        # predict_image(jpgfile)
        meanshif(jpgfile)

    # for jpgfile in glob.iglob(os.path.join(src_dir, "*.jpg")):
        # try:

        # resize_image(jpgfile)

        # result_img, bboxes_string, mask = predict_image(jpgfile)

        # # change color of image
        # alpha = 0.9  # Contrast control
        # beta = -55  # Brightness control
        #
        # # call convertScaleAbs function
        # image = cv2.imread(jpgfile)
        # adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        #
        # delete_the_background(mask, adjusted, jpgfile)
        # # except Exception:
        # #     print("bad")

    # #resize image
    # for jpgfile in glob.iglob(os.path.join(src_dir, "*.png")):
    #     try:
    #         # resize_image(jpgfile)
    #
    #         result_img, bboxes_string, mask = predict_image(jpgfile)
    #
    #         #change color of image
    #         # alpha = 0.3  # Contrast control
    #         # beta = -5  # Brightness control
    #         #
    #         # # call convertScaleAbs function
    #         # image = cv2.imread(jpgfile)
    #         # adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    #         #
    #         # delete_the_background(mask, adjusted, jpgfile)
    #     except Exception:
    #         print("bad")