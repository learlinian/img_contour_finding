import math
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.find_contour import findContours


def correlation(arr1, arr2):
    arr_2 = arr2[np.newaxis]
    arr_1 = arr1[np.newaxis]
    arr_1_conj = np.conj(arr_1)
    arr_1_t = np.transpose(arr_1_conj)

    return np.absolute(arr_2.dot(arr_1_t))[0][0]


def measure_sim(vec1, vec2):
    return correlation(vec1, vec2) / (math.sqrt(correlation(vec1, vec1)) * math.sqrt(correlation(vec2, vec2)))


def detect_contour_fft(model_path, test_img_path):
    # load fft model
    model_spectrum = np.loadtxt(model_path, dtype="complex_")
    # load the testing image
    test_img = cv2.imread(test_img_path)

    # find contours
    contours, hierarchy = findContours(test_img_path)

    # convert cartesian coordinates into complex values
    contour_list = []
    for contour in contours:
        contour_list.append(np.asarray(contour))
    contour_list_in_complex = []
    for contour_array in contour_list:
        contour_complex = []
        for pair in contour_array:
            c_number = complex(pair[0][0], pair[0][1])
            contour_complex.append(c_number)
        contour_complex = np.array(contour_complex)
        # perform discrete fourier transform
        fft_spectrum = np.fft.fft(contour_complex)
        # fft normalization
        fft_spectrum /= fft_spectrum[1]
        truncate_contour = fft_spectrum[1:model_spectrum.size + 1]
        contour_list_in_complex.append(truncate_contour)

    # calculate similarity of spectrums
    sim_list = []
    for i, spectrum in enumerate(contour_list_in_complex):
        if(spectrum.size < model_spectrum.size):
            # if contour is too small, ignore it
            sim = 0
        else:
            sim = measure_sim(spectrum, model_spectrum)
        sim_list.append(sim)

    result_list = []
    for i, acc in enumerate(sim_list):
        # set treshold of matching accuracy of 2 spectrums
        if acc > 0.93:
            result_list.append(i)

    for i, contour in enumerate(contours):
        if i in result_list:
            # draw the contour in image and stored in for validation
            cv2.drawContours(image=test_img, contours=contour,
                             contourIdx=-1, color=(0, 0, 255), thickness=3)

    cv2.imwrite('./result/detection_result.jpg', test_img)

    # # print histogram of Accuracy and observe distribution
    # plt.hist(sim_list)
    # # print detected contours
    # plt.imshow(test_img)
    # plt.show()


if __name__ == "__main__":
    detect_contour_fft('./model/truncated_fft_contour.txt', './image/a3.pgm')
