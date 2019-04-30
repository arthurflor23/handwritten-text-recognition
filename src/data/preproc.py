"""Preprocess image to the model"""

from tensorflow.image import resize, rot90, per_image_standardization
from tensorflow.keras.preprocessing import sequence, image
import numpy as np
import cv2


def process_image(img_path, nb_features):
    """Image: remove cursive style, resize and normalization"""

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = remove_cursive_style(img)

    img = np.reshape(img, img.shape + (1,))
    img = per_image_standardization(img)
    img = resize(img, size=(nb_features, 3500), preserve_aspect_ratio=True, antialias=True)
    img = rot90(img, k=3)

    image.array_to_img(img).show()

    return img[:,:,0]


def padding_list(inputs, value=1.):
    """Full fills lists with pad value"""

    return sequence.pad_sequences(inputs, value=float(value), dtype='float32', padding="post", truncating='post')


def remove_cursive_style(img):
    """Remove cursive writing style from image with deslanting algorithm"""

    def calc_y_alpha(vec):
        indices = np.where(vec > 0)[0]
        h_alpha = len(indices)

        if h_alpha > 0:
            delta_y_alpha = indices[h_alpha - 1] - indices[0] + 1

            if h_alpha == delta_y_alpha:
                return h_alpha * h_alpha
        return 0

    ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # bi_img = sauvola(img, [25, 25], 127, 0.02) if ret > 127 else otsu
    bi_img = otsu

    rows, cols = bi_img.shape
    alpha_vals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    results = []

    for alpha in alpha_vals:
        shift_x = max(-alpha * rows, 0.)
        size = (cols + int(np.ceil(abs(alpha * rows))), rows)
        transform = np.array([[1, alpha, shift_x], [0, 1, 0]], dtype=np.float)

        shear_img = cv2.warpAffine(bi_img, transform, size, cv2.INTER_NEAREST)

        sum_alpha = 0
        sum_alpha += np.apply_along_axis(calc_y_alpha, 0, shear_img)
        results.append([np.sum(sum_alpha), size, transform])

    result = sorted(results, key=lambda x: x[0], reverse=True)[0]
    return cv2.warpAffine(img, result[2], result[1], borderValue=255)


def sauvola(img, window, thresh, k):
    """Sauvola binarization method"""

    rows, cols = img.shape
    pad = int(np.floor(window[0] / 2))
    sum2, sqsum = cv2.integral2(
        cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT))

    isum = sum2[window[0]:rows + window[0], window[1]:cols + window[1]] + \
        sum2[0:rows, 0:cols] - \
        sum2[window[0]:rows + window[0], 0:cols] - \
        sum2[0:rows, window[1]:cols + window[1]]

    isqsum = sqsum[window[0]:rows + window[0], window[1]:cols + window[1]] + \
        sqsum[0:rows, 0:cols] - \
        sqsum[window[0]:rows + window[0], 0:cols] - \
        sqsum[0:rows, window[1]:cols + window[1]]

    ksize = window[0] * window[1]
    mean = isum / ksize
    std = (((isqsum / ksize) - (mean**2) / ksize) / ksize) ** 0.5
    threshold = (mean * (1 + k * (std / thresh - 1))) * (mean >= 100)

    return np.array(255 * (img >= threshold), 'uint8')
