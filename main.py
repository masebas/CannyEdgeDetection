import cv2 as cv2
import numpy as np
import scipy.stats as st


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def apply_kernel(kernel, _img):
    out = np.zeros(_img.shape, dtype='uint8')
    filter_size = len(kernel)
    kernel_size = int(np.floor(filter_size/2))

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if kernel_size <= x < _img.shape[1] - kernel_size and kernel_size <= y < _img.shape[0] - kernel_size:
                roi = _img[y-kernel_size:y+kernel_size+1, x-kernel_size:x+kernel_size+1]
                aoe = np.multiply(roi, kernel)
                out[y, x] = np.sum(a=aoe)
    return out


def gaussian_filter(filter_size, sig, img):
    if filter_size % 2 == 0:
        filter_size = filter_size-1

    out = np.zeros(img.shape, dtype='uint8')

    gauss_filter = gkern(filter_size, sig)
    print(gauss_filter)
    kernel_size = int(np.floor(filter_size/2))

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            #print(x, y)
            if kernel_size <= x < img.shape[1] - kernel_size and kernel_size <= y < img.shape[0] - kernel_size:
                roi = img[y-kernel_size:y+kernel_size+1, x-kernel_size:x+kernel_size+1]
                #print(roi.shape)
                out[y, x] = round(np.average(a=roi, weights=gauss_filter))

    cv2.imshow("output", out)
    cv2.waitKey(0)

def sobel_filter():
    print()


IMAGEPATH = "test.bmp"
sobel_vertical = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_horizontal = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

img = cv2.imread(IMAGEPATH, cv2.IMREAD_GRAYSCALE)


cv2.imshow("og", img)
out = apply_kernel(sobel_vertical, img)
cv2.imshow("Output", out)

