"""
Data preproc functions:
    augmentation: apply variations to a list of images
    normalization: apply normalization and variations on images (if required)
    encode_ctc: encode batch of texts in sparse array with padding
    standardize_texts: standardize batch of texts
    preproc: main function to the preprocess.
        Make the image:
            illumination_compensation: apply illumination regularitation
            remove_cursive_style: remove cursive style from image (if necessary)
            sauvola: apply sauvola binarization
"""

import unicodedata
import numpy as np
import string
import cv2


def augmentation(imgs,
                 dilate_range=1,
                 erode_range=1,
                 height_shift_range=0,
                 rotation_range=0,
                 scale_range=0,
                 width_shift_range=0):
    """Apply variations to a list of images (rotate, width and height shift, scale, erode, dilate)"""

    imgs = imgs.astype(np.float32)
    _, h, w = imgs.shape

    dilate_kernel = np.ones((int(np.random.uniform(1, dilate_range)),), np.uint8)
    erode_kernel = np.ones((int(np.random.uniform(1, erode_range)),), np.uint8)
    height_shift = np.random.uniform(-height_shift_range, height_shift_range)
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - scale_range, 1)
    width_shift = np.random.uniform(-width_shift_range, width_shift_range)

    trans_map = np.float32([[1, 0, width_shift * w], [0, 1, height_shift * h]])
    rot_map = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)

    trans_map_aff = np.r_[trans_map, [[0, 0, 1]]]
    rot_map_aff = np.r_[rot_map, [[0, 0, 1]]]
    affine_mat = rot_map_aff.dot(trans_map_aff)[:2, :]

    for i in range(len(imgs)):
        imgs[i] = cv2.warpAffine(imgs[i], affine_mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=255)
        imgs[i] = cv2.erode(imgs[i], erode_kernel, iterations=1)
        imgs[i] = cv2.dilate(imgs[i], dilate_kernel, iterations=1)

    return imgs


def normalization(imgs):
    """Normalize list of images"""

    imgs = imgs.astype(np.float32)
    _, h, w = imgs.shape

    for i in range(len(imgs)):
        m, s = cv2.meanStdDev(imgs[i])
        imgs[i] = imgs[i] - m[0][0]
        imgs[i] = imgs[i] / s[0][0] if s[0][0] > 0 else imgs[i]

    return np.expand_dims(imgs, axis=-1)


def decode_ctc(texts, charset):
    """Decode sparse array (sparse to text)"""

    decoded = []

    for i in range(len(texts)):
        text = "".join([charset[int(c)] for c in texts[i]])
        decoded.append(" ".join(text.split()))

    return decoded


def encode_ctc(texts, charset, max_text_length):
    """Encode text array (text to sparse)"""

    pad_encoded = np.zeros((len(texts), max_text_length))

    for i in range(len(texts)):
        texts[i] = unicodedata.normalize("NFKD", texts[i]).encode("ASCII", "ignore").decode("ASCII")
        texts[i] = " ".join(texts[i].split())

        encoded = [float(charset.find(x)) for x in texts[i] if charset.find(x) > -1]
        encoded = [float(charset.find("&"))] if len(encoded) == 0 else encoded

        pad_encoded[i, 0:len(encoded)] = encoded

    return pad_encoded


def standardize_texts(texts):
    """Organize/add spaces around punctuation marks"""

    for i in range(len(texts)):
        texts[i] = " ".join(texts[i].split()).replace(" '", "'").replace("' ", "'")
        texts[i] = texts[i].replace("«", "").replace("»", "")

        for y in texts[i]:
            if y in string.punctuation.replace("'", ""):
                texts[i] = texts[i].replace(y, f" {y} ")

        texts[i] = " ".join(texts[i].split())

    return texts


"""
Preprocess metodology based in:
    H. Scheidl, S. Fiel and R. Sablatnig,
    Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm, in
    16th International Conference on Frontiers in Handwriting Recognition, pp. 256-258, 2018.
"""


def preproc(img, img_size):
    """Make the process with the `img_size` to the scale resize"""

    wt, ht, _ = img_size
    h, w = img.shape
    f = max((w / wt), (h / ht))
    new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, new_size)

    img = illumination_compensation(img)
    img = remove_cursive_style(img)

    target = np.ones([ht, wt], dtype=np.uint8) * 255
    target[0:new_size[1], 0:new_size[0]] = img
    img = cv2.transpose(target)

    return img


"""
Illumination Compensation based in:
    K.-N. Chen, C.-H. Chen, C.-C. Chang,
    Efficient illumination compensation techniques for text images, in
    Digital Signal Processing, 22(5), pp. 726-733, 2012.
"""


def illumination_compensation(img):
    """Illumination compensation technique for text image"""

    def scale(img):
        s = np.max(img) - np.min(img)
        res = img / s
        res -= np.min(res)
        res *= 255
        return res

    img = img.astype(np.float32)
    height, width = img.shape
    sqrt_hw = np.sqrt(height * width)

    bins = np.arange(0, 300, 10)
    bins[26] = 255
    hp = np.histogram(img, bins)

    for i in range(len(hp[0])):
        if hp[0][i] > sqrt_hw:
            hr = i * 10
            break

    np.seterr(divide='ignore', invalid='ignore')
    cei = (img - (hr + 50 * 0.3)) * 2
    cei[np.where(cei > 255)] = 255
    cei[np.where(cei < 0)] = 0

    m1 = np.array([-1,0,1,-2,0,2,-1,0,1]).reshape((3,3))
    m2 = np.array([-2,-1,0,-1,0,1,0,1,2]).reshape((3,3))
    m3 = np.array([-1,-2,-1,0,0,0,1,2,1]).reshape((3,3))
    m4 = np.array([0,1,2,-1,0,1,-2,-1,0]).reshape((3,3))

    eg1 = np.abs(cv2.filter2D(img, -1, m1))
    eg2 = np.abs(cv2.filter2D(img, -1, m2))
    eg3 = np.abs(cv2.filter2D(img, -1, m3))
    eg4 = np.abs(cv2.filter2D(img, -1, m4))

    eg_avg = scale((eg1 + eg2 + eg3 + eg4) / 4)

    h, w = eg_avg.shape
    eg_bin = np.zeros((h, w))
    eg_bin[np.where(eg_avg >= 30)] = 255

    h, w = cei.shape
    cei_bin = np.zeros((h, w))
    cei_bin[np.where(cei >= 60)] = 255

    h, w = eg_bin.shape
    tli = 255 * np.ones((h,w))
    tli[np.where(eg_bin == 255)] = 0
    tli[np.where(cei_bin == 255)] = 0

    kernel = np.ones((3,3), np.uint8)
    erosion = cv2.erode(tli, kernel, iterations=1)
    int_img = np.array(cei)

    for y in range(width):
        for x in range(height):

            if erosion[x][y] == 0:
                i = x
                while(i < erosion.shape[0] and erosion[i][y] == 0):
                    i += 1

                end = i - 1
                n = end - x + 1

                if n <= 30:
                    h, e = [], []
                    for k in range(5):
                        if x - k >= 0:
                            h.append(cei[x - k][y])
                        if end + k < cei.shape[0]:
                            e.append(cei[end + k][y])

                    mpv_h, mpv_e = np.max(h), np.max(e)

                    for m in range(n):
                        int_img[x + m][y] = mpv_h + (m + 1) * ((mpv_e - mpv_h) / n)
                x = end
                break

    mean_filter = 1 / 121 * np.ones((11,11), np.uint8)
    ldi = cv2.filter2D(scale(int_img), -1, mean_filter)

    result = np.divide(cei, ldi) * 260
    result[np.where(erosion != 0)] *= 1.5
    result[result < 0] = 0
    result[result > 255] = 255

    return np.array(result, dtype=np.uint8)


"""
Deslating image process based in,
    A. Vinciarelli and J. Luettin,
    A New Normalization Technique for Cursive Handwritten Wrods, in
    Pattern Recognition, 22, 2001.
"""


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

    alpha_vals = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]
    rows, cols = img.shape
    results = []

    ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = otsu if ret < 127 else sauvola(img, (int(img.shape[0] / 2), int(img.shape[0] / 2)), 127, 1e-2)

    for alpha in alpha_vals:
        shift_x = max(-alpha * rows, 0.)
        size = (cols + int(np.ceil(abs(alpha * rows))), rows)
        transform = np.array([[1, alpha, shift_x], [0, 1, 0]], dtype=np.float)

        shear_img = cv2.warpAffine(binary, transform, size, cv2.INTER_NEAREST)
        sum_alpha = 0
        sum_alpha += np.apply_along_axis(calc_y_alpha, 0, shear_img)
        results.append([np.sum(sum_alpha), size, transform])

    result = sorted(results, key=lambda x: x[0], reverse=True)[0]
    warp = cv2.warpAffine(img, result[2], result[1], borderValue=255)

    return cv2.resize(warp, dsize=(cols, rows))


"""
Sauvola binarization based in,
    J. Sauvola, T. Seppanen, S. Haapakoski, M. Pietikainen,
    Adaptive Document Binarization, in IEEE Computer Society Washington, 1997.
"""


def sauvola(img, window, thresh, k):
    """Sauvola binarization"""

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
