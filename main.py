import cv2 as cv2
import numpy as np
import scipy.stats as st
import scipy.constants as const
from scipy import ndimage


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


def apply_kernel(kernel, _img):
    out = np.zeros(_img.shape, dtype=np.float64)
    filter_size = len(kernel)
    kernel_size = int(np.floor(filter_size/2))

    for y in range(_img.shape[0]):
        for x in range(_img.shape[1]):
            if kernel_size <= x < _img.shape[1] - kernel_size and kernel_size <= y < _img.shape[0] - kernel_size:
                roi = _img[y-kernel_size:y+kernel_size+1, x-kernel_size:x+kernel_size+1]
                aoe = np.multiply(roi, kernel)
                out[y, x] = np.sum(a=aoe)/4
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
            if kernel_size <= x < img.shape[1] - kernel_size and kernel_size <= y < img.shape[0] - kernel_size:
                roi = img[y-kernel_size:y+kernel_size+1, x-kernel_size:x+kernel_size+1]
                out[y, x] = round(np.average(a=roi, weights=gauss_filter))

    return out


def sobel_filter(img):
    horizontal_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    vertical_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_horizontal = apply_kernel(horizontal_kernel, img)
    img_vertical = apply_kernel(vertical_kernel, img)

    return img_horizontal, img_vertical


def directionalGradients(horizontal, vertical):
    magnitude = np.sqrt(np.power(horizontal, 2) + np.power(vertical, 2))
    print(magnitude)
    angle = np.arctan2(vertical, horizontal)


    return magnitude, angle


def non_maximum_suppression(mag, ang):
    width, height = mag.shape
    out = np.zeros((width, height), dtype=np.int8)
    angle = ang * 180. / np.pi

    angle[angle < 0] += 180

    for i in range(1, width-1):
        for j in range(1, height-1):
            try:
                q = 255
                r = 255

                if(0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = mag[i, j+1]
                    r = mag[i, j-1]
                elif(22.5 <= angle[i, j] < 67.5):
                    q = mag[i+1, j-1]
                    r = mag[i-1, j+1]
                elif(67.5 <= angle[i,j] < 112.5):
                    q = mag[i+1, j]
                    r = mag[i-1, j]
                elif(112.5 <= angle[i, j] < 157.5):
                    q = mag[i-1, j-1]
                    r = mag[i+1, j+1]

                if(mag[i, j] >= q) and (mag[i, j] >= r):
                    out[i, j] = mag[i, j]
                else:
                    out[i, j] = 0

            except IndexError as e:
                pass

    return out





def show_image(title, img):
    img = np.absolute(img).astype('uint8')
    cv2.imshow(title, img)

def show_image_angle(title, img):
    img = 180 + np.rad2deg(img)
    img = np.absolute((img/360)*255).astype('uint8')
    cv2.imshow(title, img)


IMAGEPATH = "test.bmp"


input = cv2.imread(IMAGEPATH, cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", input)

gauss_out = gaussian_filter(3, const.golden_ratio, input)
cv2.imshow("Post Gauss", gauss_out)

sobel_horizontal, sobel_vertical = sobel_filter(gauss_out)

show_image("Sobel Vertical", sobel_vertical)
show_image("Sobel Horizontal", sobel_horizontal)


mag, dir = directionalGradients(sobel_horizontal, sobel_vertical)
show_image("Magnitude", mag)
show_image_angle("Angle", dir)

suppressed = non_maximum_suppression(mag, dir)
show_image("NMS", suppressed)

cv2.waitKey(0)

