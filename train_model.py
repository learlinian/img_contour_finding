import cv2
import numpy as np
from utils.find_contour import findContours


def train_fft_model(img_path):
    # load image
    image = cv2.imread(img_path)
    # find contours
    contours, hierarchy = findContours(img_path)

    # extract the template contour of number '6'
    valid_contour_list = []
    for i, contour in enumerate(contours):
        # append all valid contours
        valid_contour_list.append(np.asarray(contour))
        # draw the contour in image and stored it for validation
        cv2.drawContours(image=image, contours=contour,
                             contourIdx=-1, color=(0, 0, 255), thickness=3)
    cv2.imwrite('./result/template_contour.jpg', image)
    valid_template = valid_contour_list[0]

    # convert cartesian coordinates into complex values
    contour_in_complex = []
    for pair in valid_template:
        c_number = complex(pair[0][0], pair[0][1])
        contour_in_complex.append(c_number)
    contour_in_complex = np.array(contour_in_complex)
    np.savetxt('./model/contour_complex.txt', contour_in_complex)

    # perform discrete fourier transform
    fft_spectrum = np.fft.fft(contour_in_complex)
    # fft normalization
    fft_spectrum /= abs(fft_spectrum[1])

    np.savetxt('./model/fft_contour.txt', fft_spectrum)

    # truncate the spectrum and only retain low frequency componenet (10%)
    truncated_fft_spectrum = fft_spectrum[1:int(fft_spectrum.size * 0.10)]  # remove DC component(0-frequency)

    np.savetxt('./model/truncated_fft_contour.txt', truncated_fft_spectrum)


if __name__ == "__main__":
    train_fft_model('./image/template.png')
