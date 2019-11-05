import cv2
import numpy as np
from matplotlib import pyplot as plt


def findContours(img_path):
    # read the image
    image = cv2.imread(img_path)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # blur the image using Guassian
    blur = cv2.GaussianBlur(src=gray, ksize=(13, 13), sigmaX=0)
    (t, binary) = cv2.threshold(src=blur, thresh=155, maxval=255, type=cv2.THRESH_BINARY)

    # Sobel edge detector
    sobel_x = cv2.Sobel(binary, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(binary, cv2.CV_32F, 0, 1)
    # print(sobel_x, sobel_y)
    edge_image = cv2.magnitude(sobel_x, sobel_y)
    # print(edge_image)

    edge_image = cv2.normalize(
        edge_image, None, 0., 255., cv2.NORM_MINMAX, cv2.CV_8U)

    # thining
    # edge_image = cv2.ximgproc.thinning(
    #     edge_image, None, cv2.ximgproc.THINNING_GUOHALL)

    # get contour
    cv2_version = cv2.__version__.split('.')[0]
    if cv2_version == '3':
        ret, contours, hierarchy = cv2.findContours(
            image=edge_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(
            image=edge_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    # delete the unwanted contour in the left bottom corner
    contour_length = []  # array to record contour length
    for contour in contours:
        contour_length.append(len(contour))
    del contours[contour_length.index(max(contour_length))]
    del contour_length[contour_length.index(max(contour_length))]

    contour_list = []   # list to append all contours in list

    last_i = -2     # set last valid contour index as -2
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] == -1:
            # if current contour is followed by last one, skip
            if i == last_i + 1:
                continue
            # draw the contour in image and stored in for validation
            contour_list.append(np.asarray(contour))
            last_i = i

    return contour_list, hierarchy


if __name__ == "__main__":
    contours, hierarchy = findContours(r'..\image\template.png')
    print("testing the model usability......\n")
    print(len(contours), "contours found\n")
    print("hierarchy:\n", hierarchy)
    for i, contour in enumerate(contours):
        print("\ncontour ", i + 1, " :\n")
        # print(contour)
