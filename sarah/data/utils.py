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
        if not isinstance(image, np.ndarray):
            outputs.append(image)
            continue

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
            R = 128.0
            k = 0.1

            image = image.astype(np.float32)

            mean = cv2.boxFilter(src=image, ddepth=-1, ksize=(window_size, window_size))
            mean_sq = cv2.boxFilter(src=image ** 2, ddepth=-1, ksize=(window_size, window_size))

            stddev = np.sqrt(np.maximum(mean_sq - mean ** 2, 0.0))
            threshold = mean * (1.0 + k * (stddev / R - 1.0))

            image = np.where(image > threshold, 255, 0)

        image = np.uint8(image)

        if invert:
            image = cv2.bitwise_not(image)

        outputs.append(image)

    return outputs


def batch_graph(batch_size, num_chars, graph_data, graph_shape):
    """
    Builds a batch of graph data from character graph fonts.

    Parameters
    ----------
    batch_size : int
        Number of samples in the batch.
    num_chars : int
        Number of characters per sample.
    graph_data : list
        List of graph representations from each character font.
    graph_shape : tuple or list
        The shape of the input graphs.

    Returns
    -------
    tuple
        Nodes and edges as lists.
    """

    num_fonts = len(graph_data)

    if not num_fonts:
        return [[], []]

    nodes_shape, edges_shape = graph_shape
    max_nodes = nodes_shape[0] // num_chars
    max_edges = edges_shape[1] // num_chars

    nodes = []
    edges = []

    for _ in range(batch_size):
        font_index = np.random.randint(num_fonts)
        sample_nodes = []
        sample_edges = []

        for i in range(num_chars):
            graph = graph_data[font_index][i]
            graph_nodes = graph['nodes']
            graph_edges = graph['edges']

            if len(graph_nodes) == 0:
                sample_nodes.append(np.zeros((0, 4), dtype=np.float32))
                sample_edges.append(np.zeros((2, 0), dtype=np.int32))
                continue

            n = min(graph_nodes.shape[0], max_nodes)
            e = min(graph_edges.shape[1], max_edges)

            sample_nodes.append(graph_nodes[:n].astype(np.float32))
            sample_edges.append(graph_edges[:, :e].astype(np.int32))

        nodes.append(sample_nodes)
        edges.append(sample_edges)

    return [nodes, edges]


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
        tli = np.uint8(tli) * 255
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
        image = np.uint8(image)

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

    if not batch_data:
        return batch_data

    if batch_mode == 'binary':
        batch_data = batch_padding(batch_data, target_shape=padding_shape, dtype=np.uint8)
        batch_data = np.expand_dims(batch_data, axis=-1)

        if batch_scale:
            batch_data = (batch_data.astype(np.float32) / 255.)

    elif batch_mode == 'image':
        if illumination:
            batch_data = batch_illumination(batch_data)

        if binarization:
            batch_data = batch_binarization(batch_data, method=binarization)

        batch_data = batch_padding(batch_data, target_shape=padding_shape, dtype=np.uint8)
        batch_data = np.expand_dims(batch_data, axis=-1)

        if batch_scale:
            batch_data = (batch_data.astype(np.float32) / 127.5) - 1

    elif batch_mode == 'graph':
        nodes, edges = batch_data

        if not nodes:
            return [nodes, edges]

        batch_size = len(nodes)
        num_chars = len(nodes[0])
        nodes_shape, edges_shape = padding_shape

        max_nodes = nodes_shape[0] // num_chars
        max_edges = edges_shape[1] // num_chars

        pad_nodes = np.zeros((batch_size, num_chars, max_nodes, 4), dtype=np.float32)
        pad_edges = np.zeros((batch_size, num_chars, 2, max_edges), dtype=np.int32)

        for i in range(batch_size):
            for j in range(num_chars):
                char_nodes = nodes[i][j].copy()
                char_edges = edges[i][j].copy()

                if char_nodes.shape[0] == 0:
                    continue

                n = min(char_nodes.shape[0], max_nodes)
                e = min(char_edges.shape[1], max_edges)

                pad_nodes[i, j, :n] = char_nodes[:n]
                pad_edges[i, j, :, :e] = char_edges[:, :e] + j * max_nodes

        pad_nodes = pad_nodes.reshape(batch_size, max_nodes * num_chars, 4)
        pad_edges = pad_edges.reshape(batch_size, 2, max_edges * num_chars)

        batch_data = [pad_nodes, pad_edges]

    elif batch_mode == 'text':
        batch_data = batch_padding(batch_data, target_shape=padding_shape, dtype=np.int64)

    return batch_data


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


def plot_graph(coordinates, nodes, edges, target_shape=None):
    """
    Plot graph nodes and edges.

    Parameters
    ----------
    coordinates : ndarray or tf.Tensor
        Node coordinates.
    nodes : ndarray or tf.Tensor
        Node features.
    edges : ndarray or tf.Tensor
        Edge index pairs.
    target_shape : tuple, optional
        Target image shape.

    Returns
    -------
    ndarray
        Rendered graph image.
    """

    if hasattr(coordinates, 'numpy'):
        coordinates = coordinates.numpy()

    if hasattr(nodes, 'numpy'):
        nodes = nodes.numpy()

    if hasattr(edges, 'numpy'):
        edges = edges.numpy()

    coordinates = np.asarray(coordinates)
    nodes = np.asarray(nodes)
    edges = np.asarray(edges)

    valid = nodes[:, 0] != 0
    num_nodes = np.count_nonzero(valid)
    coordinates = coordinates[:num_nodes]

    valid_edges = (
        (edges[0] < num_nodes)
        & (edges[1] < num_nodes)
        & (edges[0] != edges[1])
    )

    edges = edges[:, valid_edges]

    if target_shape is not None:
        height, width = target_shape[:2]

    elif num_nodes > 0:
        width = int(np.max(coordinates[:, 0])) + 1
        height = int(np.max(coordinates[:, 1])) + 1

    else:
        height, width = 1, 1

    canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

    for u, v in edges.T:
        p1 = tuple(coordinates[u].astype(int))
        p2 = tuple(coordinates[v].astype(int))

        cv2.line(canvas, p1, p2, (40, 40, 200), 1)

    for x, y in coordinates:
        x, y = int(x), int(y)
        cv2.rectangle(canvas, (x - 1, y - 1), (x + 1, y + 1), (0, 0, 220), -1)

    return canvas


def plot_graph_font(nodes, edges, chars, cols=None, cell_size=80, skip=3):
    """
    Plot a character grid from graph font nodes and edges.

    Parameters
    ----------
    nodes : ndarray or list
        Node coordinates for characters.
    edges : ndarray or list
        Edge index pairs for characters.
    chars : list
        Character labels for each graph.
    cols : int, optional
        Number of grid columns.
    cell_size : int, optional
        Pixel size of each character cell.
    skip : int, optional
        Number of leading characters to skip.

    Returns
    -------
    ndarray
        Rendered grid image.
    """

    num_chars = len(chars)
    padded = isinstance(nodes, np.ndarray)

    if padded:
        max_nodes = nodes.shape[0] // num_chars
        max_edges = edges.shape[1] // num_chars
        pad_edges = edges.reshape(num_chars, 2, max_edges)

    indices = [
        i for i in range(skip, num_chars)
        if (np.any(nodes[i * max_nodes:(i + 1) * max_nodes, 0] != 0)
            if padded else len(nodes[i]) > 0)
    ]

    num_plot = len(indices)
    cols = cols or int(np.ceil(np.sqrt(num_plot)))
    rows = int(np.ceil(num_plot / cols))

    width = cols * cell_size
    height = rows * cell_size

    canvas = np.ones((height, width), dtype=np.uint8) * 255
    pad = 10
    inner = cell_size - 2 * pad

    for i, j in enumerate(indices):
        char = chars[j]

        if padded:
            node_start = j * max_nodes
            char_nodes = nodes[node_start:node_start + max_nodes]
            char_edges = pad_edges[j]
            valid = char_nodes[:, 0] != 0
        else:
            node_start = 0
            char_nodes = nodes[j]
            char_edges = edges[j]
            valid = np.ones(len(char_nodes), dtype=bool)

        max_local_nodes = len(char_nodes)

        row, col = divmod(i, cols)
        x0 = col * cell_size
        y0 = row * cell_size

        px = x0 + pad + char_nodes[:, 0] * inner
        py = y0 + pad + (1 - char_nodes[:, 1]) * inner
        pts = np.stack([px, py], axis=1).astype(np.int32)

        for s, d in char_edges.T:
            s, d = int(s) - node_start, int(d) - node_start

            if s == d or not (0 <= s < max_local_nodes and 0 <= d < max_local_nodes):
                continue
            if not (valid[s] and valid[d]):
                continue

            cv2.line(canvas, tuple(pts[s]), tuple(pts[d]), 0, 2, cv2.LINE_AA)

        cv2.putText(canvas, char, (x0 + 2, y0 + cell_size - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1, cv2.LINE_AA)

    return canvas


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

    return np.uint8(image)


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

    return np.uint8(image)
