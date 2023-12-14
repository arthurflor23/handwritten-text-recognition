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


def prepare_text_batch(text_batch, target_shape=None, pad_value=0, dtype=np.int32):
    """
    Pads a batch of text data to a uniform shape.

    Parameters
    ----------
    text_batch : list
        List of texts.
    target_shape : tuple, optional
        Target shape for padding.
    pad_value : int, optional
        Value used for padding.
    dtype : data-type, optional
        Data type of the output array.

    Returns
    -------
    numpy.ndarray
        Padded text data.
    """

    if target_shape:
        max_paragraphs, max_lines, max_chars = target_shape[:3]
    else:
        max_paragraphs = max(len(text) for text in text_batch)
        max_lines = max(len(paragraph) for text in text_batch for paragraph in text)
        max_chars = max(len(line) for text in text_batch for paragraph in text for line in paragraph)

    padded = np.full((len(text_batch), max_paragraphs, max_lines, max_chars), pad_value, dtype=dtype)

    for i, text in enumerate(text_batch):
        paragraphs, lines, chars = np.asarray(text).shape
        padded[i, :paragraphs, :lines, :chars] = text

    text_batch = np.expand_dims(padded, axis=-1)

    return text_batch


def prepare_image_batch(image_batch, target_shape=None, pad_value=255, dtype=np.uint8):
    """
    Pads and processes a batch of image data to a uniform shape.

    Parameters
    ----------
    image_batch : list
        List of image arrays.
    target_shape : list or tuple, optional
        Target shape for padding.
    pad_value : int, optional
        Value used for padding.
    dtype : data-type, optional
        Data type of the output array.

    Returns
    -------
    numpy.ndarray
        Padded and processed image data.
    """

    if target_shape:
        max_height, max_width = target_shape[:2]
    else:
        max_height = max(len(height) for height in image_batch)
        max_width = max(len(width) for image in image_batch for width in image)

    padded = np.full((len(image_batch), max_height, max_width), pad_value, dtype=dtype)

    for i, image in enumerate(image_batch):
        height, width = np.asarray(image).shape
        padded[i, :height, :width] = image

    image_batch = np.expand_dims(padded, axis=-1)
    image_batch = image_batch.transpose((0, 2, 1, 3))
    image_batch = (image_batch.astype(np.float32) / 127.5) - 1

    return image_batch


def read_image(image_path, bbox=None, image_shape=None):
    """
    Read an image from the given file path and perform optional bbox.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    bbox : list, optional
        The bbox coordinates ([x, y, width, height]).
    image_shape : list or tuple, optional
        Image shape for resizing.

    Returns
    -------
    ndarray
        The loaded image as a NumPy array.
    """

    if not isinstance(image_path, str):
        return image_path

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

    if image_shape is not None:
        image = resize_image(image, target_shape=image_shape)

    return image


def resize_image(image, target_shape):
    """
    Resize the image to fit within the target shape, maintaining the aspect ratio.

    Parameters
    ----------
    image : ndarray
        Input image.
    target_shape : list or tuple
        Target shape.

    Returns
    -------
    ndarray
        Resized image.
    """

    if not target_shape:
        return image

    h, w = image.shape
    target_w, target_h = target_shape[:2]

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
