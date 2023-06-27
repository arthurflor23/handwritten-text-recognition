import os
import re
import cv2
import html
import string
import importlib
import concurrent
import numpy as np


class Dataset():
    """
    General data source management.
    """

    def __init__(self,
                 source=None,
                 level=None,
                 training_ratio=None,
                 validation_ratio=None,
                 test_ratio=None,
                 pad_value=255,
                 lazy_mode=True,
                 infer_data=None,
                 artifact_path='data',
                 seed=None):
        """
        Initializes a new instance of the Dataset class.

        Parameters
        ----------
        source : str, optional
            The data source name. Default is None.
        level : str, optional
            The recognition level. Default is None.
        training_ratio : float or int, optional
            The training ratio for resample. Default is None.
        validation_ratio : float or int, optional
            The validation ratio for resample. Default is None.
        test_ratio : float or int, optional
            The test ratio for resample. Default is None.
        pad_value : int, optional
            Padding value. Default is 255.
        lazy_mode : bool, optional
            Lazy mode flag for lazy loading process. Default is True.
        infer_data : list, optional
            Custom data for inference mode. Default is None.
        artifact_path : str, optional
            Path name to fetch the data. Default is 'data'.
        seed : int, optional
            The random seed. Default is None.

        Returns
        -------
        None
        """

        np.random.seed(seed)

        self.source = source
        self.level = level
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.pad_value = pad_value
        self.lazy_mode = lazy_mode
        self.artifact_path = artifact_path
        self.seed = seed

        self.size = 0
        self.corpus = ''
        self.charset = []

        self.min_text = ''
        self.max_text = ''

        self.min_rows = np.inf
        self.max_rows = -np.inf

        self.min_cols = np.inf
        self.max_cols = -np.inf

        if infer_data is None:
            self._source = self._import_source(self.source)
            self._source = self._source(self.artifact_path)

            data = self._source.fetch_data(self.level)
            data = self._prepare_source_data(data, infer=False)
        else:
            data = self._prepare_source_data(infer_data, infer=True)

        self.training = self._create_partition(data[0], test=False)
        self.validation = self._create_partition(data[1], test=False)
        self.test = self._create_partition(data[2], test=True)

        self.tokenizer = Tokenizer(self.charset, self.max_rows, self.max_cols)

        self.training = self._encode_partition_labels(self.training)
        self.validation = self._encode_partition_labels(self.validation)
        self.test = self._encode_partition_labels(self.test)

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        attributes = {
            'source': self.source,
            'level': self.level,
            'training_ratio': self.training_ratio,
            'validation_ratio': self.validation_ratio,
            'test_ratio': self.test_ratio,
            'pad_value': self.pad_value,
            'lazy_mode': self.lazy_mode,
            'seed': self.seed,
            'size': self.size,
            'charset': ''.join(self.charset),
            'max_rows': self.max_rows,
            'max_cols': self.max_cols,
        }

        return attributes

    def __str__(self):
        """
        Returns a string representation of the object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        info = f"""
            Dataset Configuration\n
            Source                  {self.source or '-'}
            Recognition Level       {self.level or '-'}
            Training Ratio          {self.training_ratio or '-'}
            Validation Ratio        {self.validation_ratio or '-'}
            Test Ratio              {self.test_ratio or '-'}
            Padding Value           {self.pad_value}
            Lazy Mode               {self.lazy_mode}
            Seed                    {self.seed}

            \nDataset Information\n
            Total Size              {self.size}

            Training Size           {self.training['size']}
            Validation Size         {self.validation['size']}
            Test Size               {self.test['size']}

            Corpus Length           {len(self.corpus)}
            Corpus                  {self.corpus[:100]} [...]

            Charset Length          {len(self.charset)}
            Charset                 {''.join(self.charset)}

            Min Text Length         {len(self.min_text)}
            Min Text                {self.min_text}

            Max Text Length         {len(self.max_text)}
            Max Text                {self.max_text}

            Min Rows                {self.min_rows}
            Max Rows                {self.max_rows}

            Min Columns             {self.min_cols}
            Max Columns             {self.max_cols}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def get_generator(self,
                      partition,
                      batch_size=16,
                      augmentor=None,
                      raw_data=False,
                      shuffle=True):
        """
        Generates a batch of data samples for the specified partition.

        Parameters
        ----------
        partition : dict
            The dataset partition which will be create the generator.
        batch_size : int, optional
            The number of samples in each batch, default is 16.
        augmentor : Augmentor, optional
            The Augmentor class. Default is None.
        raw_data : bool, optional
            Specifies whether to generate raw or processed data, default is False.
        shuffle : bool, optional
            Specifies whether shuffles per epoch, default is True.

        Returns
        -------
        tuple
            A generator for data batches and steps per epoch.
        """

        def generator(partition, subset, indices):
            batch_index = 0

            while True:
                if batch_index >= partition['size']:
                    if shuffle:
                        np.random.shuffle(indices)
                    batch_index = 0

                batch_indices = indices[batch_index:batch_index + batch_size]
                batch_index += batch_size

                batch_data = partition[subset][batch_indices]

                x_data = batch_data[:, 0]
                y_data = batch_data[:, 2]

                if self.lazy_mode:
                    x_data = [self.read_image(data[0], data[1]) for data in batch_data]

                if augmentor:
                    x_data = [augmentor.augmentation(x, x_data) for x in x_data]

                if not raw_data:
                    x_data = self._pad_batch_data(x_data, self.pad_value, np.uint8)
                    y_data = self._pad_batch_data(y_data, self.tokenizer.pad_tk_index, np.int32)

                yield (x_data, y_data)

        subset = 'raw' if raw_data else 'data'
        indices = np.arange(partition['size'])

        batch_generator = generator(partition, subset, indices)
        steps_per_epoch = np.math.ceil(partition['size'] / batch_size)

        return batch_generator, steps_per_epoch

    def read_image(self, image_path, bbox=None):
        """
        Read an image from the given file path and perform optional bbox.

        Parameters
        ----------
        image_path : str
            The path to the image file.
        bbox : tuple, optional
            The bbox coordinates (x, y, width, height). Default is None.

        Returns
        -------
        numpy.ndarray
            The loaded image as a NumPy array.
        """

        if not isinstance(image_path, str):
            return image_path

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if bbox is None or len(bbox) != 4:
            return image

        x, y, width, height = bbox

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

    def _import_source(self, source):
        """
        Dynamically imports the specified source.

        Parameters
        ----------
        source : str
            The name of the source to be imported.

        Returns
        -------
        source : instance of Source class
            An instance of the Source class from the imported module.

        Raises
        ------
        AssertionError
            If the specified source file or Source class don't exist.
        """

        module_name = importlib.util.resolve_name(f".source.{source}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "source file must be created"

        module = importlib.import_module(module_name, __package__)

        class_name = 'Source'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        source = getattr(module, class_name)

        return source

    def _prepare_source_data(self, data, infer=False):
        """
        Prepares the data for partitioning.

        Parameters
        ----------
        data : tuple
            The input data (training, validation, test).
        infer : bool, optional
            Flag for inference mode. Default is False.

        Returns
        -------
        list
            The prepared data.
        """

        data = list(data)

        if not 1 <= len(data) <= 3:
            raise ValueError("input data must have 1 to 3 partition lists")

        if infer:
            data = [[], [], data]

        for i in range(len(data)):
            if not data[i]:
                continue

            data[i] = list(data[i]) or []
            data[i] = list(map(list, data[i]))

            for j in range(len(data[i])):
                data[i][j] = list(data[i][j]) or []

                if not data[i][j]:
                    continue

                if not 1 <= len(data[i][j]) <= 3:
                    raise ValueError("partition item must have 1 to 3 dims [images, bbox (opt), labels (opt)]")

                if len(data[i][j]) == 1:
                    data[i][j] = [data[i][j][0] or '', [], ['']]

                elif len(data[i][j]) == 2:
                    if isinstance(data[i][1], str):
                        data[i][j] = [data[i][j][0] or '', [], data[i][j][1] or '']
                    else:
                        data[i][j] = [data[i][j][0] or '', data[i][j][1] or [], ['']]

                elif len(data[i][j]) == 3:
                    data[i][j] = [data[i][j][0] or '', data[i][j][1] or [], data[i][j][2] or '']

                if len(data[i][j][1]) != 0 and len(data[i][j][1]) != 4:
                    raise ValueError("bbox value must have 0 or 4 dims [x, y, width, height]")

                data[i][j].extend([[]] * (3 - len(data[i][j])))

            data[i].extend([[]] * (3 - len(data[i])))

        data.extend([[]] * (3 - len(data)))

        ratios = [self.training_ratio, self.validation_ratio, self.test_ratio]
        ratio_is_not_none = [ratio for ratio in ratios if ratio is not None]

        if ratio_is_not_none:
            for i in range(len(ratios)):
                if not ratios[i]:
                    continue

                if isinstance(ratios[i], str):
                    ratios[i] = float(ratios[i]) if '.' in ratios[i] else int(ratios[i])

            ratio = sum(x for x in ratios if x is not None)

            if isinstance(ratio, float) and ratio == 1.0:
                merged = []

                for i, ratio in enumerate(ratios):
                    if ratio is not None:
                        np.random.shuffle(data[i])
                        merged.extend(data[i])

                if merged:
                    total_merged = len(merged)

                    for i, ratio in enumerate(ratios):
                        if ratio is not None:
                            np.random.shuffle(merged)
                            index = round((ratio + 1e-8) * total_merged)
                            data[i] = merged[:index]
                            merged = merged[index:]

            else:
                for i, ratio in enumerate(ratios):
                    if ratio is not None:
                        np.random.shuffle(data[i])
                        index = round((ratio + 1e-8) * len(data[i])) if isinstance(ratio, float) else ratio
                        data[i] = data[i][:index]

        for i in range(len(data)):
            if not data[i]:
                continue

            np.random.shuffle(data[i])

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._unpack_data_item, x) for x in data[i]]
                data[i] = [list(x) for x in [future.result() for future in futures] if x]

        return data

    def _unpack_data_item(self, item, infer=False):
        """
        Load and validate data item.

        Parameters
        ----------
        item : tuple
            The data item to validate.
        infer : bool, optional
            Flag for inference mode. Default is False.

        Returns
        -------
        tuple
            The loaded and validated data item.
        """

        image_path, bbox, label = item
        image = None

        if os.path.exists(image_path) and os.path.isfile(image_path):
            try:
                image = self.read_image(image_path, bbox)

            except Exception:
                print(f"Image `{os.path.basename(image_path)}` cannot be read.")
                return None
        else:
            print(f"Image `{os.path.basename(image_path)}` does not exist or is not a file.")
            return None

        if image is None or image.size == 0:
            print(f"Image `{os.path.basename(image_path)}` has an invalid size.")
            return None

        label = self._format_text(label)

        if not infer and not label:
            print(f"Image `{os.path.basename(image_path)}` has an invalid label.")
            return None

        image = image_path if self.lazy_mode else image

        return [image_path, bbox, label], [image, bbox, label]

    def _format_text(self, text):
        """
        Clean and format the input text.

        Parameters
        ----------
        text : str
            The input text to be cleaned.

        Returns
        -------
        str
            The formatted text.
        """

        if isinstance(text, str):
            text = text.split('\n')

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

            line = re.sub(f'([{re.escape(string.punctuation)}])', r' \1 ', line)
            line = re.sub(r'\s+', ' ', line).strip()

            text[i] = line

        return text

    def _create_partition(self, partition_data, test=False):
        """
        Creates a partition dictionary from the given partition data.

        Parameters
        ----------
        partition_data : tuple
            The partition data containing labels, images, and bbox information.
        test : bool, optional
            If set to True, the function will handle the partition data as test data. Default is False.

        Returns
        -------
        dict
            The partition dict.
        """

        dct = {
            'raw': [],
            'data': [],
            'size': 0,
            'corpus': '',
            'charset': [],
            'min_text': '',
            'max_text': '',
            'min_rows': 0,
            'max_rows': 0,
            'min_cols': 0,
            'max_cols': 0,
        }

        dct['raw'] = np.array([x[0] for x in partition_data], dtype=object)
        dct['data'] = np.array([x[1] for x in partition_data], dtype=object)
        dct['size'] = dct['data'].shape[0]

        labels = [x[2] for x in dct['data'] if x[2]]

        if labels:
            dct['corpus'] = ' '.join(' '.join(x) for x in labels).strip()
            dct['charset'] = sorted(set(''.join(''.join(x) for x in labels)))
            dct['min_text'] = min(['\\n'.join(x) for x in labels], key=len)
            dct['max_text'] = max(['\\n'.join(x) for x in labels], key=len)
            dct['min_rows'] = min(len(x) for x in labels)
            dct['max_rows'] = max(len(x) for x in labels)
            dct['min_cols'] = min(len(y) for x in labels for y in x)
            dct['max_cols'] = max(len(y) for x in labels for y in x)

        self.size += dct['size']

        if not test:
            self.corpus = f"{self.corpus} {dct['corpus']}".strip()
            self.charset = sorted(set(self.charset + dct['charset']))

        if dct['min_text']:
            self.min_text = min(self.min_text or dct['min_text'], dct['min_text'], key=len)

        if dct['max_text']:
            self.max_text = max(self.max_text or dct['max_text'], dct['max_text'], key=len)

        if dct['min_rows']:
            self.min_rows = min(self.min_rows, dct['min_rows'])

        if dct['max_rows']:
            self.max_rows = max(self.max_rows, dct['max_rows'])

        if dct['min_cols']:
            self.min_cols = min(self.min_cols, dct['min_cols'])

        if dct['max_cols']:
            self.max_cols = max(self.max_cols, dct['max_cols'])

        return dct

    def _encode_partition_labels(self, partition):
        """
        Encode labels in the partition data using the tokenizer.

        Parameters
        ----------
        partition : dict
            A dictionary containing 'data', a list of lists,
            where the last element of each sublist is a label to be encoded.

        Returns
        -------
        partition : dict
            The same partition dictionary with 'data' updated to contain encoded labels
            and converted to a numpy array of dtype object.
        """

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.tokenizer.encode, x[-1]) for x in partition['data']]
            encoded_labels = [future.result() for future in futures]

            for i in range(len(partition['data'])):
                partition['data'][i][-1] = encoded_labels[i]

        return partition

    def _pad_batch_data(self, batch_data, pad_value=255, dtype=None):
        """
        Pads each 2D sub-array in the batch data to the maximum height and width.

        Parameters
        ----------
        data : list
            List of 2D sub-arrays to be padded.
        pad_value : int, optional
            Padding value. Default is 255.
        dtype : data-type, optional
            Desired data type of output array.

        Returns
        -------
        numpy.ndarray
            Padded batch data.
        """

        max_height = max(len(data) for data in batch_data)
        max_width = max(len(item) for data in batch_data for item in data)

        padded = np.full((len(batch_data), max_height, max_width), pad_value, dtype=dtype)

        for i, data in enumerate(batch_data):
            for j, item in enumerate(data):
                padded[i, j, :len(item)] = item

        padded = np.expand_dims(padded, axis=-1)

        return padded


class Tokenizer():
    """
    Class for tokenizing data using a character set.
    """

    def __init__(self, charset, max_rows, max_cols):
        """
        Initialize the Tokenizer.

        Parameters
        ----------
        charset : list
            List of characters in the character set.
        max_rows : int, optional
            Maximum number of rows for each label.
        max_cols : int, optional
            Maximum number of columns for each label.
        """

        self.pad_tk = '¶'
        self.unk_tk = '◬'

        self.charset = [self.pad_tk, self.unk_tk] + charset
        self.shape = (max_rows, max_cols + (len(self.charset) - len(charset)), len(self.charset) + 1)

        self.pad_tk_index = self.charset.index(self.pad_tk)
        self.unk_tk_index = self.charset.index(self.unk_tk)

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        attributes = {
            'charset': ''.join(self.charset),
            'shape': str(self.shape),
        }

        return attributes

    def __str__(self):
        """
        Returns a string representation of the object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        info = f"""
            Tokenizer Configuration\n
            Charset             {self.charset}
            Charset Length      {len(self.charset)}
            Shape               {self.shape}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def encode(self, label):
        """
        Encode a single label by mapping characters to their corresponding token indices.

        Parameters
        ----------
        label : str
            Label to encode.

        Returns
        -------
        list
            Encoded label with token indices.
        """

        encoded_label = []

        for row in label:
            enconded_row = []

            for char in row:
                index = self.charset.index(char) if char in self.charset else self.unk_tk_index
                enconded_row.append(index)

            encoded_label.append(enconded_row)

        return encoded_label

    def decode(self, encoded_label):
        """
        Decode a single encoded label by converting token indices back to characters.

        Parameters
        ----------
        encoded_label : list
            Encoded label with token indices.

        Returns
        -------
        list
            Decoded label with characters.
        """

        label = []

        for enconded_row in encoded_label:
            row = ''

            for enconded_char in enconded_row:
                if int(enconded_char) == -1:
                    continue

                row += self.charset[int(enconded_char)]

            row = row.replace(self.pad_tk, '')
            row = row.replace(self.unk_tk, '')

            label.append(row)

        return label
