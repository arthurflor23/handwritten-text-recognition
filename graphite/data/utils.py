import re
import cv2
import html


def read_image(image_path, bbox=None, image_shape=None):
    """
    Read an image from the given file path and perform optional bbox.

    Parameters
    ----------
    image_path : str
        The path to the image file.
    bbox : list, optional
        The bbox coordinates ([x, y, width, height]).
    image_shape : list, optional
        Image shape for resizing.

    Returns
    -------
    ndarray
        The loaded image as a NumPy array.
    """

    if not isinstance(image_path, str):
        return image_path

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

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
    target_shape : list
        Target shape as [height, width, channels].

    Returns
    -------
    ndarray
        Resized image.
    """

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

    # image = image.reshape((image.shape[0], image.shape[1], 1))

    return image


def format_text(text, breakline=False):
    """
    Clean and format the input text.

    Parameters
    ----------
    text : str
        The input text to be cleaned.
    breakline : bool, optional
        Break lines of the input data.

    Returns
    -------
    str
        The formatted text.
    """

    if isinstance(text, str):
        text = text.split('\n') if breakline else [text]

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

    for i, line in enumerate(text):
        line = html.unescape(line)

        for pattern, replacement in regexes.items():
            line = pattern.sub(replacement, line)

        text[i] = re.sub(r'\s+', ' ', line).strip()

    if not breakline:
        text = text[0]

    return text
