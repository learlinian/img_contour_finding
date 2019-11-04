import cv2
import numpy as np


if __name__ == "__main__":
    # read the image
    image = cv2.imread('a3.pgm')

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
    cv_version = cv2.__version__.split('.')[0]
    if cv_version == '3':
        ret, contours, hierarchy = cv2.findContours(
            image=edge_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    else:
        contours, hierarchy = cv2.findContours(
            image=edge_image, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    print(type(contours))
    contour_length = []  # array to record contour length
    for contour in contours:
        contour_length.append(len(contour))

    # delete the the longest contour in the corner
    del contours[contour_length.index(max(contour_length))]
    del contour_length[contour_length.index(max(contour_length))]

    # store contour info
    with open('contour.txt', 'w+') as f:
        for contour in contours:
            f.write(str(contour))

    # draw the contour in image
    for i, contour in enumerate(contours):
        if hierarchy[0, i, 3] == -1:
            cv2.drawContours(image=image, contours=contour,
                             contourIdx=-1, color=(0, 0, 255), thickness=3)
    cv2.imwrite('contour.jpg', image)
