import cv2

# read the image
image = cv2.imread('a3.pgm')
gray = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2GRAY)

# blur the image using Guassian
blur = cv2.GaussianBlur(src=gray, ksize=(5, 5), sigmaX=0)
(t, binary) = cv2.threshold(src=blur, thresh=120, maxval=255, type=cv2.THRESH_BINARY)

# get contour
contours, _ = cv2.findContours(image=binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

contour_length = []  # array to record contour length
for contour in contours:
    contour_length.append(len(contour))

del contours[contour_length.index(max(contour_length))]  # delete the the longest contour in the corner
del contour_length[contour_length.index(max(contour_length))]

# store contour info
with open('contour.txt', 'w+') as f:
    for contour in contours:
        f.write(str(contour))

# draw the contour in image
for contour in contours:
    cv2.drawContours(image=image, contours=contour, contourIdx=-1, color=(0, 0, 255), thickness=3)
cv2.imwrite('contour.jpg', image)
