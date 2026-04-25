import re
import cv2
import html
import numpy as np


def batch_binarization(batch_data, method, invert=False):
    """
    Apply binarization to a batch of grayscale images.

    Parameters
    ----------
    batch_data : list of np.ndarray
        List of grayscale images.
    method : str, optional
        Binarization method to apply.
    invert : bool, optional
        Whether to invert the binarized image.

    Returns
    ----------
    list of np.ndarray
        List of binarized images.
    """

    outputs = []

    for image in batch_data:
        if method == 'global':
            _, image = cv2.threshold(src=image,
                                     thresh=127,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY)

        elif method == 'otsu':
            _, image = cv2.threshold(src=image,
                                     thresh=0,
                                     maxval=255,
                                     type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        elif method == 'adaptive_mean':
            image = cv2.adaptiveThreshold(src=image,
                                          maxValue=255,
                                          adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,
                                          thresholdType=cv2.THRESH_BINARY,
                                          blockSize=11,
                                          C=2)

        elif method == 'adaptive_gaussian':
            image = cv2.adaptiveThreshold(src=image,
                                          maxValue=255,
                                          adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          thresholdType=cv2.THRESH_BINARY,
                                          blockSize=11,
                                          C=2)

        elif method == 'sauvola':
            window_size = 31
            thresh = 128
            k = 0.1

            mean = cv2.boxFilter(src=image, ddepth=-1, ksize=(window_size, window_size))
            mean_sq = cv2.boxFilter(src=image**2, ddepth=-1, ksize=(window_size, window_size))

            stddev = np.sqrt(mean_sq - mean**2)
            threshold = mean * (1 + k * (stddev / thresh - 1))

            image = np.where(image > threshold, 255, 0).astype(np.uint8)

        if invert:
            image = cv2.bitwise_not(image)

        outputs.append(image)

    return outputs


def batch_illumination(batch_data):
    """
    Apply illumination compensation to enhance text visibility in images.

    Parameters
    ----------
    batch_data : list of np.ndarray
        List of grayscale images.

    Returns
    -------
    list of np.ndarray
        List of processed images.

    References
    ----------
    Efficient illumination compensation techniques for text images.
        https://www.sciencedirect.com/science/article/pii/S1051200412000826
    """

    outputs = []

    for image in batch_data:
        _, binary = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)

        if np.sum(binary) > np.sum(image) * 0.8:
            outputs.append(image.copy())
            continue

        imgf = image.astype(np.float32)
        h, w = image.shape

        hist = cv2.calcHist([image], [0], None, [26], [0, 260])
        hr = np.argmax(hist > np.sqrt(image.size)) * 10

        cei = np.clip((imgf - (hr + 15)) * 2, 0, 255)

        gx = cv2.Sobel(imgf, cv2.CV_32F, 1, 0, 3)
        gy = cv2.Sobel(imgf, cv2.CV_32F, 0, 1, 3)

        mag = cv2.magnitude(gx, gy)

        rng = mag / (mag.max() - mag.min())
        rng = (rng - rng.min()) * 255.

        tli = ~((rng >= 30) | (cei >= 60))
        tli = tli.astype(np.uint8) * 255
        erosion = cv2.erode(tli, np.ones((3, 3), np.uint8), 1)

        img = cei.copy()
        for y in range(w):
            mask = erosion[:, y] == 0

            if not np.any(mask):
                continue

            diff = np.diff(mask.astype(np.int8))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0]

            if mask[0]:
                starts = np.r_[0, starts]

            if mask[-1]:
                ends = np.r_[ends, h - 1]

            for s, e in zip(starts, ends):
                n = e - s + 1

                if n > 30:
                    continue

                top = cei[max(0, s - 5):s, y]
                bot = cei[e + 1:min(h, e + 6), y]

                if top.size == 0 or bot.size == 0:
                    continue

                img[s:e + 1, y] = np.linspace(top.max(), bot.max(), n)

        rng = img / (img.max() - img.min())
        rng = (rng - rng.min()) * 255.

        rng = cv2.blur(rng, (11, 11))

        image = np.nan_to_num(cei / rng) * 260
        image[erosion != 0] *= 1.5

        image = np.clip(image, 0, 255)
        image = image.astype(np.uint8)

        outputs.append(image)

    return outputs


def batch_masking(batch_data,
                  max_shape,
                  char_height=64,
                  char_width=16,
                  from_text=False):
    """
    Generate image masks for a batch.

    Parameters
    ----------
    batch_data : ndarray
        Batch of data (images or texts).
    max_shape : tuple
        Maximum shape for each mask.
    char_height : int, optional
        Pixel height per character row.
    char_width : int, optional
        Pixel width per character column.
    from_text : bool, optional
        Whether to generate masks from images or text lengths.

    Returns
    -------
    list of ndarray
        Item masks indicating content areas.
    """

    max_height, max_width = max_shape[:2]
    masks = []

    for item in batch_data:
        if isinstance(item, str):
            height, width = max_height, max_width

        elif from_text:
            height = len(item) * char_height
            width = max(len(row) for row in item) * char_width

        else:
            height, width = item.shape

        shape = (min(height, max_height), min(width, max_width))
        masks.append(np.full(shape=shape, fill_value=255, dtype=np.uint8))

    return masks


def batch_padding(batch_data, pad_value=0, target_shape=None, dtype=np.int64):
    """
    Pads a batch of data to a uniform shape.

    Parameters
    ----------
    batch_data : list
        List of data.
    pad_value : int, optional
        Value used for padding.
    target_shape : tuple, optional
        Target shape for padding.
    dtype : data-type, optional
        Data type of the output array.

    Returns
    -------
    numpy.ndarray
        Padded data.
    """

    if target_shape:
        max_height, max_width = target_shape[:2]
    else:
        max_height = max(len(height) for height in batch_data)
        max_width = max(len(width) for height in batch_data for width in height)

    shape = (len(batch_data), max_height, max_width)
    padded = np.full(shape=shape, fill_value=pad_value, dtype=dtype)

    for i, data in enumerate(batch_data):
        data = np.array(data)

        if data.size > 0 and data.ndim == 2:
            padded[i, :data.shape[0], :data.shape[1]] = data

    return padded


def batch_processing(batch_mode,
                     batch_data,
                     batch_scale=True,
                     padding_shape=None,
                     illumination=False,
                     binarization=None):
    """
    Processes a data batch for model input.

    Parameters
    ----------
    batch_mode : str
        Type of input data.
    batch_data : list
        List of data input.
    batch_scale : bool, optional
        Whether to scale data values.
    padding_shape : tuple, optional
        Target shape for padding.
    illumination : bool, optional
        Apply illumination compensation.
    binarization : str, optional
        Apply binarization method.

    Returns
    -------
    numpy.ndarray
        Processed data.
    """

    if batch_mode in ['image', 'binary']:
        if batch_mode == 'image':
            if illumination:
                batch_data = batch_illumination(batch_data)

            if binarization:
                batch_data = batch_binarization(batch_data, method=binarization)

        batch_data = batch_padding(batch_data, target_shape=padding_shape, dtype=np.uint8)
        batch_data = np.expand_dims(batch_data, axis=-1)

        if batch_scale:
            if batch_mode == 'image':
                batch_data = (batch_data.astype(np.float32) / 127.5) - 1
            else:
                batch_data = (batch_data.astype(np.float32) / 255.)

    elif batch_mode == 'text':
        batch_data = batch_padding(batch_data, target_shape=padding_shape, dtype=np.int64)

    return np.array(batch_data)


def format_text(text):
    """
    Clean and format the input text.

    Parameters
    ----------
    text : str
        The input text to be cleaned.

    Returns
    -------
    str or list
        The formatted text.
    """

    substitutions = {
        r'[ ]': ' ',
        r'[＿]': '_',
        r'[，]': ',',
        r'[；]': ';',
        r'[：]': ':',
        r'[！﹗]': '!',
        r'[？﹖]': '?',
        r'[．。]': '.',
        r'[＂“”″‶]': '"',
        r'[（]': '(',
        r'[）]': ')',
        r'[［]': '[',
        r'[］]': ']',
        r'[｛]': '}',
        r'[｝]': '{',
        r'[＠]': '@',
        r'[＊]': '*',
        r'[／]': '/',
        r'[＼]': '\\\\',
        r'[＆]': '&',
        r'[＃]': '#',
        r'[％]': '%',
        r'[＾]': '^',
        r'[˗֊‐‑‒–—－−﹣]': '-',
        r'[＋]': '+',
        r'[＜]': '<',
        r'[＝]': '=',
        r'[＞]': '>',
        r'[｜]': '|',
        r'[～]': '~',
        r'[⋯]': '...',
        r'[＄]': '$',
        r"[＇ʼ´‘’‛′‵`᾽᾿՚׳❛❜｀`]": '\'',
    }

    regexes = {re.compile(pattern): replacement for pattern, replacement in substitutions.items()}

    for pattern, replacement in regexes.items():
        text = pattern.sub(replacement, text)

    text = html.unescape(text)
    text = ' '.join(text.replace('\n', '﹗').split()).replace('﹗', '\n').strip()

    return text


def read_image(image_path, bbox=None):
    """
    Read an image from the given file path and perform optional bbox.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    bbox : list, optional
        The bbox coordinates ([x, y, width, height]).

    Returns
    -------
    ndarray
        The loaded image as a NumPy array.
    """

    if not image_path:
        return np.full((1, 1), fill_value=0, dtype=np.uint8)

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        return image

    if bbox is not None and len(bbox) == 4:
        x, y, width, height = bbox

        if isinstance(x, str):
            x = float(x) if '.' in x else int(x)

        if isinstance(y, str):
            y = float(y) if '.' in y else int(y)

        if isinstance(width, str):
            width = float(width) if '.' in width else int(width)

        if isinstance(height, str):
            height = float(height) if '.' in height else int(height)

        if isinstance(x, float):
            x = int(x * image.shape[1])

        if isinstance(y, float):
            y = int(y * image.shape[0])

        if isinstance(width, float):
            width = int(width * image.shape[1])

        if isinstance(height, float):
            height = int(height * image.shape[0])

        y = max(0, abs(y - 10))
        x = max(0, abs(x - 10))

        height = min(image.shape[0], (height + 10))
        width = min(image.shape[1], (width + 10))

        image = image[y:y+height, x:x+width]

    return np.array(image, dtype=np.uint8)


def resize_image(image, target_width=None, target_shape=None):
    """
    Resize the image to fit within the target shape, maintaining the aspect ratio.

    Parameters
    ----------
    image : ndarray
        Input image.
    target_width : int, optional
        Target image width from character widths.
    target_shape : list or tuple
        Target image shape.

    Returns
    -------
    ndarray
        Resized image.
    """

    if image is None or image.size <= 1:
        return None

    src_h, src_w = image.shape[:2]

    if target_width and target_shape:
        new_h = target_shape[0]
        new_w = min(target_width, target_shape[1])
        interpolation = cv2.INTER_CUBIC if new_h > src_h or new_w > src_w else cv2.INTER_AREA

        image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=interpolation)

    elif target_shape:
        dst_h, dst_w = target_shape[:2]

        if src_h > dst_h or src_w > dst_w:
            aspect_ratio = src_w / src_h

            if aspect_ratio < 1:
                new_h = min(dst_h, int(dst_w / aspect_ratio))
                new_w = int(new_h * aspect_ratio)
            else:
                new_w = min(dst_w, int(dst_h * aspect_ratio))
                new_h = int(new_w / aspect_ratio)

            interpolation = cv2.INTER_CUBIC if new_h > src_h or new_w > src_w else cv2.INTER_AREA

            image = cv2.resize(src=image, dsize=(new_w, new_h), interpolation=interpolation)

    return np.array(image, dtype=np.uint8)
