"""Uses generator functions to supply train/test with data.
Image renderings and text are created on the fly each time"""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import h5py
import cv2


class DataGenerator():
    """Generator class with data streaming"""

    def __init__(self, env):

        if env.worker_mode:
            with h5py.File(env.source, "r") as hf:
                self.dataset = {
                    "train": {
                        "dt": hf["train"]["dt"][:],
                        "gt": hf["train"]["gt"][:]
                    },
                    "valid": {
                        "dt": hf["valid"]["dt"][:],
                        "gt": hf["valid"]["gt"][:]
                    },
                    "test": {
                        "dt": hf["test"]["dt"][:],
                        "gt": hf["test"]["gt"][:]
                    }
                }
        else:
            self.dataset = h5py.File(env.source, "r")

        self.max_text_length = env.max_text_length
        self.batch_size = max(2, env.batch_size)
        self.train_index, self.valid_index, self.test_index = 0, 0, 0

        self.generator = ImageDataGenerator(
            fill_mode="constant",
            rotation_range=0.5,
            width_shift_range=0.02,
            height_shift_range=0.02,
            zoom_range=0.02
        )

        self.total_train = self.dataset["train"]["gt"][:].shape[0]
        self.total_valid = self.dataset["valid"]["gt"][:].shape[0]
        self.total_test = self.dataset["test"]["gt"][:].shape[0]

        self.train_steps = self.total_train // self.batch_size
        self.valid_steps = self.total_valid // self.batch_size
        self.test_steps = self.total_test // self.batch_size

    def fill_batch(self, partition, total, x, y):
        """Fill batch array (x, y) if necessary (batch_size)"""

        if len(x) < self.batch_size:
            fill = self.batch_size - len(x)
            i = np.random.choice(np.arange(0, total - fill), 1)[0]
            x = np.append(x, self.dataset[partition]["dt"][i:i + fill], axis=0)
            y = np.append(y, self.dataset[partition]["gt"][i:i + fill], axis=0)
        return x, y

    def next_train_batch(self):
        """Get the next batch from train partition (yield)"""

        while True:
            if self.train_index >= self.total_train:
                self.train_index = 0

            index = self.train_index
            until = self.train_index + self.batch_size
            self.train_index += self.batch_size

            x_train = self.dataset["train"]["dt"][index:until]
            y_train = self.dataset["train"]["gt"][index:until]
            x_train, y_train = self.fill_batch("train", self.total_train, x_train, y_train)

            x_train_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_train_len = np.asarray([len(np.trim_zeros(y_train[i])) for i in range(self.batch_size)])

            x_train = self.generator.flow(x_train, batch_size=self.batch_size, shuffle=False)[0]

            inputs = {
                "input": x_train,
                "labels": y_train,
                "input_length": x_train_len,
                "label_length": y_train_len
            }
            output = {"CTCloss": np.zeros(self.batch_size)}

            yield (inputs, output)

    def next_valid_batch(self):
        """Get the next batch from validation partition (yield)"""

        while True:
            if self.valid_index >= self.total_valid:
                self.valid_index = 0

            index = self.valid_index
            until = self.valid_index + self.batch_size
            self.valid_index += self.batch_size

            x_valid = self.dataset["valid"]["dt"][index:until]
            y_valid = self.dataset["valid"]["gt"][index:until]
            x_valid, y_valid = self.fill_batch("valid", self.total_valid, x_valid, y_valid)

            x_valid_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_valid_len = np.asarray([len(np.trim_zeros(y_valid[i])) for i in range(self.batch_size)])

            inputs = {
                "input": x_valid,
                "labels": y_valid,
                "input_length": x_valid_len,
                "label_length": y_valid_len
            }
            output = {"CTCloss": np.zeros(self.batch_size)}

            yield (inputs, output)

    def next_test_batch(self):
        """Return model evaluate parameters"""

        while True:
            if self.test_index >= self.total_test:
                self.test_index = 0

            index = self.test_index
            until = self.test_index + self.batch_size
            self.test_index += self.batch_size

            x_test = self.dataset["test"]["dt"][index:until]
            y_test = self.dataset["test"]["gt"][index:until]
            x_test, y_test = self.fill_batch("test", self.total_test, x_test, y_test)

            x_test_len = np.asarray([self.max_text_length for i in range(self.batch_size)])
            y_test_len = np.asarray([len(np.trim_zeros(y_test[i])) for i in range(self.batch_size)])

            yield [x_test, y_test, x_test_len, y_test_len]


"""
Data preproc functions:
    encode_ctc: encode batch of texts in sparse array with padding
    decode_ctc: dencode batch arrays in text
    preproc: main function to the preprocess.
        Make the image:
            illumination_compensation: apply illumination regularitation
            remove_cursive_style: remove cursive style from image (if necessary)
            sauvola: apply sauvola binarization
"""


def encode_ctc(text, charset, mtl):
    """Encode text batch to CTC input (sparse)"""

    pad_encoded = np.zeros(mtl)
    encoded = [[np.abs(float(charset.find(x))) for x in vec][0] for vec in text.strip()]
    pad_encoded[0:len(encoded)] = encoded

    return pad_encoded


def decode_ctc(arr, charset):
    """Decode CTC output batch to text"""

    return np.array([("".join(charset[int(c)] for c in vec)).strip() for vec in arr])


"""
Preprocess metodology based in:
    H. Scheidl, S. Fiel and R. Sablatnig,
    Word Beam Search: A Connectionist Temporal Classification Decoding Algorithm, in
    16th International Conference on Frontiers in Handwriting Recognition, pp. 256-258, 2018.
"""


def preproc(img, img_size, read_first=False):
    """Make the process with the `img_size` to the scale resize"""

    if read_first:
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE)

    wt, ht, _ = img_size
    h, w = img.shape
    f = max((w / wt), (h / ht))
    new_size = (max(min(wt, int(w / f)), 1), max(min(ht, int(h / f)), 1))
    img = cv2.resize(img, new_size)

    img = illumination_compensation(img)
    img = remove_cursive_style(img)

    target = np.ones([ht, wt])
    target[0:new_size[1], 0:new_size[0]] = (img / 255)
    img = cv2.transpose(target)

    m, s = cv2.meanStdDev(img)
    img = img - m[0][0]
    img = img / s[0][0] if s[0][0] > 0 else img

    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    return np.reshape(img, img.shape + (1,))


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

    kernel = np.ones((3,3),np.uint8)
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
