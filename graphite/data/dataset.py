import os
import re
import cv2
import html
import string
import random
import zipfile
import importlib
import numpy as np
import concurrent.futures


class Dataset():
    """
    General data source management.
    """

    def __init__(self,
                 source=None,
                 text_level=None,
                 image_shape=None,
                 training_ratio=None,
                 validation_ratio=None,
                 test_ratio=None,
                 binarization=False,
                 lazy_mode=False,
                 data=None,
                 artifact_path='dataset',
                 seed=None):
        """
        Initializes the Dataset class.

        Parameters
        ----------
        source : str, optional
            The data source name. Default is None.
        text_level : str, optional
            The text structure level. Default is None.
        image_shape : list, optional
            The images shape. Default is None.
        training_ratio : float or int, optional
            The training ratio for resample. Default is None.
        validation_ratio : float or int, optional
            The validation ratio for resample. Default is None.
        test_ratio : float or int, optional
            The test ratio for resample. Default is None.
        binarization : bool, optional
            Apply binarization method. Default is False.
        lazy_mode : bool, optional
            Enable lazy loading mode. Default is False.
        data : list, optional
            Data for inference mode. Default is None.
        artifact_path : str, optional
            Path name to fetch the data. Default is 'dataset'.
        seed : int, optional
            Seed for random shuffle. Default is None.
        """

        random.seed(seed)
        np.random.seed(seed)

        self.source = source
        self.text_level = text_level
        self.image_shape = tuple(image_shape)
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.binarization = binarization
        self.lazy_mode = lazy_mode
        self.artifact_path = artifact_path
        self.seed = seed

        self.size = 0
        self.corpus = ''
        self.tokens = []
        self.charset = []

        self.min_rows = np.inf
        self.max_rows = -np.inf

        self.min_cols = np.inf
        self.max_cols = -np.inf

        if data is None:
            self._source = self._import_source(self.source)
            self._source = self._source(self.artifact_path)

            if hasattr(self._source, 'base_path'):
                self._extract_source_zip(self.artifact_path, self._source.base_path)

            data = self._source.fetch_data(self.text_level)
            data = self._prepare_data(data, training=True)
        else:
            data = self._prepare_data(data, training=False)

        self.partitions = ['training', 'validation', 'test']
        self.dt, self.tokenizer = self._build_partitions(data)

    def __repr__(self):
        """
        Returns a string representation of the object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        info = f"""
            ============================================
            Dataset Configuration
            --------------------------------------------
            Source                  {self.source or '-'}
            Text Level              {self.text_level or '-'}
            Image Shape             {self.image_shape or '-'}
            Training Ratio          {self.training_ratio or '-'}
            Validation Ratio        {self.validation_ratio or '-'}
            Test Ratio              {self.test_ratio or '-'}
            Binarization            {self.binarization}
            Lazy Mode               {self.lazy_mode}
            Seed                    {self.seed}
            ============================================
            Dataset Information
            --------------------------------------------
            Total Size              {self.size}

            Training Size           {self.dt['training']['size']}
            Validation Size         {self.dt['validation']['size']}
            Test Size               {self.dt['test']['size']}

            Corpus Length           {len(self.corpus)}
            Tokens Length           {len(self.tokens)}
            Charset Length          {len(self.charset)}

            Charset                 {''.join(self.charset)}

            Min Rows                {self.min_rows}
            Max Rows                {self.max_rows}

            Min Columns             {self.min_cols}
            Max Columns             {self.max_cols}
            {self.tokenizer}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()]).strip()

        return info

    # def _to_dict(self):
    #     """
    #     Convert the class object attributes to a dictionary.

    #     Returns
    #     -------
    #     dict
    #         A dictionary with the class attributes.
    #     """

    #     attributes = {
    #         'source': self.source,
    #         'text_level': self.text_level,
    #         'image_shape': self.image_shape,
    #         'training_ratio': self.training_ratio,
    #         'validation_ratio': self.validation_ratio,
    #         'test_ratio': self.test_ratio,
    #         'binarization': self.binarization,
    #         'lazy_mode': self.lazy_mode,
    #         'seed': self.seed,
    #         'size': self.size,
    #         'corpus': self.corpus,
    #         'tokens': self.tokens,
    #         'charset': self.charset,
    #         'max_rows': self.max_rows,
    #         'max_cols': self.max_cols,
    #     }

    #     return attributes

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

    def _extract_source_zip(self, artifact_path, source):
        """
        Extracts a .zip file into a directory if the directory doesn't exist yet.

        Parameters
        ----------
        source : str
            The base name of the .zip file and target directory.
        """

        if not source.startswith(artifact_path):
            source = os.path.join(artifact_path, source)

        if not os.path.exists(source) and os.path.isfile(f'{source}.zip'):
            with zipfile.ZipFile(f'{source}.zip', 'r') as zip_ref:
                zip_ref.extractall(artifact_path)

    def _prepare_data(self, data, training=False):
        """
        Prepares the data for partitioning.

        Parameters
        ----------
        data : tuple
            The input data (training, validation, test).
        training : bool, optional
            If True, the function will handle the partition as training. Default is False.

        Returns
        -------
        list
            The prepared data.
        """

        data = list(data)

        if not 1 <= len(data) <= 3:
            raise ValueError("input data must have 1 to 3 partition lists")

        if not training:
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

        if training and len(data[1]) == 0:
            np.random.shuffle(data[0])
            index = round((self.validation_ratio or 0.1) * len(data[0]))

            data[1] = data[0][:index]
            data[0] = data[0][index:]

            self.validation_ratio = None

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
                    if ratio is None:
                        continue

                    np.random.shuffle(data[i])
                    merged.extend(data[i])

                total_merged = len(merged)

                if total_merged > 0:
                    for i, ratio in enumerate(ratios):
                        if ratio is None:
                            continue

                        np.random.shuffle(merged)
                        index = round((ratio + 1e-8) * total_merged)

                        data[i] = merged[:index]
                        merged = merged[index:]

            else:
                for i, ratio in enumerate(ratios):
                    if ratio is None:
                        continue

                    np.random.shuffle(data[i])
                    index = round((ratio + 1e-8) * len(data[i])) if isinstance(ratio, float) else ratio

                    data[i] = data[i][:index]

        for i in range(len(data)):
            if not data[i]:
                continue

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._validate_data_item, x, training) for x in data[i]]
                data[i] = [list(x) for x in [future.result() for future in futures] if x]

        return data

    def _validate_data_item(self, item, training=False):
        """
        Load and validate data item.

        Parameters
        ----------
        item : tuple
            The data item to validate.
        training : bool, optional
            If True, the function will handle the partition as training. Default is False.

        Returns
        -------
        tuple
            The loaded and validated data item.
        """

        if len(item) == 0:
            return None

        image_path, bbox, label = item
        image = None

        if os.path.exists(image_path):
            try:
                image = self._read_image(image_path, self.image_shape, bbox)

            except Exception:
                print(f"Image `{os.path.basename(image_path)}` cannot be read.")
                return None
        else:
            print(f"Image `{os.path.basename(image_path)}` does not exist.")
            return None

        if image is None or image.size == 0:
            print(f"Image `{os.path.basename(image_path)}` has an invalid size.")
            return None

        label = self._format_text(label)

        if training and not label:
            print(f"Image `{os.path.basename(image_path)}` has an invalid label.")
            return None

        if self.lazy_mode:
            image = image_path

        return [image_path, bbox, label], [image, bbox, label]

    def _read_image(self, image_path, image_shape, bbox=None):
        """
        Read an image from the given file path and perform optional bbox.

        Parameters
        ----------
        image_path : str
            The path to the image file.
        image_shape : list
            Image shape for resizing.
        bbox : tuple, optional
            The bbox coordinates (x, y, width, height). Default is None.

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

            # String
            if isinstance(x, str):
                x = float(x) if '.' in x else int(x)

            if isinstance(y, str):
                y = float(y) if '.' in y else int(y)

            if isinstance(width, str):
                width = float(width) if '.' in width else int(width)

            if isinstance(height, str):
                height = float(height) if '.' in height else int(height)

            # Number
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

        image = self._resize_image(image, image_shape)

        return image

    def _resize_image(self, image, target_shape):
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

    def _build_partitions(self, data):
        """
        Build data partitions and process the data within each partition.

        Parameters
        ----------
        data : list
            A list containing data to be partitioned and processed.

        Returns
        -------
        tuple
            A tuple containing a dictionary with processed data for each partition, and a tokenizer object.
        """

        dt = {}

        for i, partition in enumerate(self.partitions):
            dt[partition] = {
                'raw_data': [],
                'data': [],
                'size': 0,
                'corpus': '',
                'tokens': [],
                'charset': [],
                'min_rows': 0,
                'max_rows': 0,
                'min_cols': 0,
                'max_cols': 0,
            }

            dt[partition]['raw_data'] = np.array([x[0] for x in data[i]], dtype=object)
            dt[partition]['data'] = np.array([x[1] for x in data[i]], dtype=object)
            dt[partition]['size'] = dt[partition]['data'].shape[0]

            labels = [x[2] for x in dt[partition]['data'] if x[2]]

            if labels:
                dt[partition]['corpus'] = ' '.join(' '.join(x) for x in labels).strip()
                dt[partition]['tokens'] = sorted(set(' '.join(' '.join(x) for x in labels).split()))
                dt[partition]['charset'] = sorted(set(''.join(''.join(x) for x in labels)))
                dt[partition]['min_rows'] = min(len(x) for x in labels)
                dt[partition]['max_rows'] = max(len(x) for x in labels)
                dt[partition]['min_cols'] = min(len(y) for x in labels for y in x)
                dt[partition]['max_cols'] = max(len(y) for x in labels for y in x)

            self.size += dt[partition]['size']

            if partition != 'test':
                self.corpus = f"{self.corpus} {dt[partition]['corpus']}".strip()
                self.tokens = sorted(set(self.tokens + dt[partition]['tokens']))
                self.charset = sorted(set(self.charset + dt[partition]['charset']))

            if dt[partition]['min_rows']:
                self.min_rows = min(self.min_rows, dt[partition]['min_rows'])

            if dt[partition]['max_rows']:
                self.max_rows = max(self.max_rows, dt[partition]['max_rows'])

            if dt[partition]['min_cols']:
                self.min_cols = min(self.min_cols, dt[partition]['min_cols'])

            if dt[partition]['max_cols']:
                self.max_cols = max(self.max_cols, dt[partition]['max_cols'])

        tokenizer = Tokenizer(self.charset, self.max_rows, self.max_cols)

        for partition in self.partitions:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(tokenizer.encode, x[-1]) for x in dt[partition]['data']]
                encoded_labels = [future.result() for future in futures]

                for i in range(len(dt[partition]['data'])):
                    dt[partition]['data'][i][-1] = encoded_labels[i]

        return dt, tokenizer

#     def get_generator(self,
#                       partition,
#                       batch_size=8,
#                       augmentor=None,
#                       raw_data=False,
#                       shuffle=True):
#         """
#         Generates a batch of data samples for the specified partition.

#         Parameters
#         ----------
#         partition : dict
#             The dataset partition which will be create the generator.
#         batch_size : int, optional
#             The number of samples in each batch, default is 8.
#         augmentor : Augmentor, optional
#             The Augmentor class. Default is None.
#         raw_data : bool, optional
#             Specifies whether to generate raw or processed data, default is False.
#         shuffle : bool, optional
#             Specifies whether shuffles per epoch, default is True.

#         Returns
#         -------
#         tuple
#             A generator for data batches and steps per epoch.
#         """

#         def generator(partition, subset, indices):
#             batch_index = 0

#             while True:
#                 if batch_index >= partition['size']:
#                     if shuffle:
#                         np.random.shuffle(indices)
#                     batch_index = 0

#                 batch_indices = indices[batch_index:batch_index + batch_size]
#                 batch_index += batch_size

#                 batch_data = partition[subset][batch_indices]

#                 x_data = batch_data[:, 0]
#                 y_data = batch_data[:, 2]

#                 if self.lazy_mode or raw_data:
#                     x_data = [self._read_image(data[0], data[1]) for data in batch_data]

#                 if self.binarization:
#                     x_data = [self._binarization(x) for x in x_data]

#                 if augmentor:
#                     x_data = [augmentor.augmentation(x, x_data) for x in x_data]

#                 if not raw_data:
#                     x_data = self._pad_batch_data(x_data, 255, np.uint8)
#                     y_data = self._pad_batch_data(y_data, self.tokenizer.pad_tk_index, np.int32)

#                 yield (x_data, y_data)

#         subset = 'raw' if raw_data else 'data'
#         indices = np.arange(partition['size'])

#         batch_generator = generator(partition, subset, indices)
#         steps_per_epoch = int(np.ceil(partition['size'] / batch_size))

#         return batch_generator, steps_per_epoch

#     def _binarization(self, image):
#         """
#         Apply binarization method to an image.

#         Parameters
#         ----------
#         image : ndarray
#             Input image to be binarized.

#         Returns
#         ----------
#         ndarray
#             Binarized image.
#         """

#         _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

#         return image

#     def _pad_batch_data(self, batch_data, pad_value, dtype=None):
#         """
#         Pads each 2D sub-array in the batch data to the maximum height and width.

#         Parameters
#         ----------
#         data : list
#             List of 2D sub-arrays to be padded.
#         pad_value : int, optional
#             Padding value.
#         dtype : data-type, optional
#             Desired data type of output array.

#         Returns
#         -------
#         ndarray
#             Padded batch data.
#         """

#         max_height = max(len(data) for data in batch_data)
#         max_width = max(len(item) for data in batch_data for item in data)

#         padded = np.full((len(batch_data), max_height, max_width), pad_value, dtype=dtype)

#         for i, data in enumerate(batch_data):
#             for j, item in enumerate(data):
#                 padded[i, j, :len(item)] = item

#         padded = np.expand_dims(padded, axis=-1)

#         return padded


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
        self.sos_tk = '◖'
        self.eos_tk = '◗'
        self.unk_tk = '◬'

        self.charset = [self.pad_tk, self.sos_tk, self.eos_tk, self.unk_tk] + charset
        self.shape = (max_rows, max_cols + (len(self.charset) - len(charset)), len(self.charset) + 1)

        self.pad_tk_index = self.charset.index(self.pad_tk)
        self.sos_tk_index = self.charset.index(self.sos_tk)
        self.eos_tk_index = self.charset.index(self.eos_tk)
        self.unk_tk_index = self.charset.index(self.unk_tk)

    def __repr__(self):
        """
        Returns a string representation of the object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        info = f"""
            ============================================
            Tokenizer Configuration
            --------------------------------------------
            Charset                 {''.join(self.charset)}
            Charset Length          {len(self.charset)}
            Shape                   {self.shape}
            ============================================
        """

        info = '\n'.join([x.strip() for x in info.splitlines()]).strip()

        return info

    # def _to_dict(self):
    #     """
    #     Convert the class object attributes to a dictionary.

    #     Returns
    #     -------
    #     dict
    #         A dictionary with the class attributes.
    #     """

    #     attributes = {
    #         'charset': self.charset,
    #         'shape': self.shape,
    #     }

    #     return attributes

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

        charset_dict = {char: index for index, char in enumerate(self.charset)}
        sos_tk_index = self.sos_tk_index
        eos_tk_index = self.eos_tk_index
        unk_tk_index = self.unk_tk_index

        def encode_row(row):
            encoded_row = [sos_tk_index]
            encoded_row.extend(charset_dict.get(char, unk_tk_index) for char in row)
            encoded_row.append(eos_tk_index)
            return encoded_row

        return [encode_row(row) for row in label]

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

        translation_table = str.maketrans('', '', self.pad_tk + self.sos_tk + self.eos_tk + self.unk_tk)

        def decode_row(encoded_row):
            row = ''.join([self.charset[int(encoded_char)] for encoded_char in encoded_row if int(encoded_char) != -1])
            return row.translate(translation_table)

        return [decode_row(encoded_row) for encoded_row in encoded_label]

    # def encode(self, label):
    #     """
    #     Encode a single label by mapping characters to their corresponding token indices.

    #     Parameters
    #     ----------
    #     label : str
    #         Label to encode.

    #     Returns
    #     -------
    #     list
    #         Encoded label with token indices.
    #     """

    #     encoded_label = []

    #     for row in label:
    #         enconded_row = [self.sos_tk_index]

    #         for char in row:
    #             index = self.charset.index(char) if char in self.charset else self.unk_tk_index
    #             enconded_row.append(index)

    #         enconded_row += [self.eos_tk_index]
    #         encoded_label.append(enconded_row)

    #     return encoded_label

    # def decode(self, encoded_label):
    #     """
    #     Decode a single encoded label by converting token indices back to characters.

    #     Parameters
    #     ----------
    #     encoded_label : list
    #         Encoded label with token indices.

    #     Returns
    #     -------
    #     list
    #         Decoded label with characters.
    #     """

    #     label = []

    #     for enconded_row in encoded_label:
    #         row = ''

    #         for enconded_char in enconded_row:
    #             if int(enconded_char) == -1:
    #                 continue

    #             row += self.charset[int(enconded_char)]

    #         row = row.replace(self.pad_tk, '')
    #         row = row.replace(self.sos_tk, '')
    #         row = row.replace(self.eos_tk, '')
    #         row = row.replace(self.unk_tk, '')

    #         label.append(row)

    #     return label
