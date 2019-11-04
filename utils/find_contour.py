import cv2
import numpy as np
from matplotlib import pyplot as plt

def findContours(img_path):
    # read the image
    image = cv2.imread(img_path)

    # grayscale
    gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

    # blur the image using Guassian
    blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
    (t, binary) = cv2.threshold(src=blur,
                                thresh=120, maxval=255, type=cv2.THRESH_BINARY)
    
    # Sobel edge detector
    sobel_x = cv2.Sobel(binary, cv2.CV_32F, 1, 0)
    sobel_y = cv2.Sobel(binary, cv2.CV_32F, 0, 1)
    edge_image = cv2.magnitude(sobel_x, sobel_y)
    edge_image = cv2.normalize(
    edge_image, None, 0., 255., cv2.NORM_MINMAX, cv2.CV_8U)

    # thining
    edge_image = cv2.ximgproc.thinning(
    edge_image, None, cv2.ximgproc.THINNING_GUOHALL)

    # get contour
    cv2_version = cv2.__version__.split('.')[0]
    if cv2_version == '3':
        ret, contours, hierarchy = cv2.findContours(
            image=edge_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(
            image=edge_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    return contours, hierarchy

if __name__ == "__main__":
    contours, hierarchy = findContours('./image/template.png')
    print("testing the model usability......\n")
    print(len(contours), "contours found\n")
    print("hierarchy:\n", hierarchy)
    for i, contour in enumerate(contours):
        print("\ncontour ", i+1, " :\n")
        # print(contour)
    