import re
import cv2
import html
import numpy as np


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


def batch_padding(batch_data, target_shape=None, pad_value=0, dtype=np.int64):
    """
    Pads a batch of data to a uniform shape.

    Parameters
    ----------
    batch_data : list
        List of data.
    target_shape : tuple, optional
        Target shape for padding.
    pad_value : int, optional
        Value used for padding.
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

    padded = np.full((len(batch_data), max_height, max_width), fill_value=pad_value, dtype=dtype)

    for i, data in enumerate(batch_data):
        data = np.asarray(data)

        if data.size > 0 and data.ndim == 2:
            padded[i, :data.shape[0], :data.shape[1]] = data

    return padded


def batch_processing(batch_data, image_processing=False):
    """
    Processes a data batch for model input .

    Parameters
    ----------
    data_batch : list
        List of arrays.
    image_processing : bool, optional
        Whether to apply image processing or not.

    Returns
    -------
    numpy.ndarray
        Processed data.
    """

    if image_processing:
        batch_data = np.expand_dims(batch_data, axis=-1)
        batch_data = (batch_data.astype(np.float32) / 127.5) - 1

    return batch_data


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

    return image


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

    if target_width and target_shape:
        new_h, new_w = target_shape[0], min(target_width, target_shape[1])
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    elif target_shape:
        h, w = image.shape
        target_h, target_w = target_shape[:2]

        if h > target_h or w > target_w:
            aspect_ratio = w / h

            if aspect_ratio >= 1:
                new_w = min(target_w, int(target_h * aspect_ratio))
                new_h = int(new_w / aspect_ratio)
            else:
                new_h = min(target_h, int(target_w / aspect_ratio))
                new_w = int(new_h * aspect_ratio)

            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image
