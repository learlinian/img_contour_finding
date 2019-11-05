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
        fft_spectrum = np.fft.fft(contour_complex)
        # fft normalization
        for i, spectrum in enumerate(fft_spectrum):
            fft_spectrum[i] = spectrum / abs(fft_spectrum[1])
        truncate_contour = fft_spectrum[1:model_spectrum.size + 1]
        contour_list_in_complex.append(truncate_contour)

    # print(contour_list_in_complex[1])

    model_spectrum_magnitude = np.absolute(model_spectrum)

    rmse_list = []
    for i, spectrum in enumerate(contour_list_in_complex):
        spectrum_magnitude = np.absolute(spectrum)
        if(spectrum_magnitude.size < model_spectrum_magnitude.size):
            extended_array = np.zeros(model_spectrum_magnitude.shape)
            extended_array[: spectrum_magnitude.shape[0]] = spectrum_magnitude
            mse = np.sqrt((extended_array - model_spectrum_magnitude)**2).mean()
        else:
            mse = np.sqrt((spectrum_magnitude - model_spectrum_magnitude)**2).mean()
        rmse_list.append(mse)
    
    # print(rmse_list[1])
    # plt.hist(rmse_list)
    # plt.show()

    result_list = []
    for i, error in enumerate(rmse_list):
        if error < 190:
            result_list.append(i)

    print(rmse_list)

    for i, contour in enumerate(contours):
        if i in result_list:
            # draw the contour in image and stored in for validation
            cv2.drawContours(image=test_img, contours=contour,
                             contourIdx=-1, color=(0, 0, 255), thickness=3)
    
    plt.imshow(test_img)
    plt.show()


    

if __name__ == "__main__":
    detect_contour_fft('./model/truncated_fft_contour.txt', './image/a3.pgm')