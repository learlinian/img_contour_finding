import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.find_contour import findContours


def detect_contour_fft(model_path, test_img_path):
    # load fft model
    model_spectrum = np.loadtxt(model_path, dtype="complex_")
    # load the testing image
    test_img = cv2.imread(test_img_path)

    # find contours
    contours, hierarchy = findContours(test_img_path)

    # delete the unwanted contour in the left bottom corner
    contour_length = []  # array to record contour length
    for contour in contours:
        contour_length.append(len(contour))
    del contours[contour_length.index(max(contour_length))]
    del contour_length[contour_length.index(max(contour_length))]

    contour_list = []
    for contour in contours:
        contour_list.append(np.asarray(contour))

    # convert cartesian coordinates into complex values
    contour_list_in_complex = []
    for contour_array in contour_list:
        contour_complex = []
        for pair in contour_array:
            c_number = complex(pair[0][0], pair[0][1])
            contour_complex.append(c_number)
        contour_complex = np.array(contour_complex)
        contour_list_in_complex.append(contour_complex)

    
    

    

if __name__ == "__main__":
    detect_contour_fft('./model/truncated_fft_contour.txt', './image/a3.pgm')