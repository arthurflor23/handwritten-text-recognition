import os
import re
import cv2
import json
import html
import nltk
import string
import importlib
import concurrent
import numpy as np
import multiprocessing


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
        self.base_path = os.path.join(os.path.dirname(__file__), '..', '..')
        self.artifact_path = os.path.join(self.base_path, artifact_path)
        self.lazy_mode = lazy_mode
        self.seed = seed

        self.size = 0
        self.corpus = ''
        self.charset = []

        self.min_text = ''
        self.max_text = ''

        self.min_rows = float('inf')
        self.max_rows = float('-inf')

        self.min_cols = float('inf')
        self.max_cols = float('-inf')

        if infer_data is None:
            self._source = self._import_source(self.source)
            self._source = self._source(self.artifact_path)

            data = self._source.fetch_data(self.level)
            data, self.reference_pixels = self._prepare_data(data, infer=False)
        else:
            data, self.reference_pixels = self._prepare_data(infer_data, infer=True)

        self.training = self._create_partition_dictionary(data[0], test=False)
        self.validation = self._create_partition_dictionary(data[1], test=False)
        self.test = self._create_partition_dictionary(data[2], test=True)

        self.tokenizer = Tokenizer(self.charset, self.max_rows, self.max_cols)

        self.training['data'] = self.tokenizer.encode_data(self.training['data'])
        self.validation['data'] = self.tokenizer.encode_data(self.validation['data'])
        self.test['data'] = self.tokenizer.encode_data(self.test['data'])

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the Dataset object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        return json.dumps({
            'source': self.source,
            'level': self.level,
            'training_ratio': self.training_ratio,
            'validation_ratio': self.validation_ratio,
            'test_ratio': self.test_ratio,
            'lazy_mode': self.lazy_mode,
            'seed': self.seed,
            'size': self.size,
            'reference_pixels': self.reference_pixels,
            'corpus': self.corpus,
            'charset': self.charset,
            'charset_length': len(self.charset),
            'min_text': self.min_text,
            'min_text_length': len(self.min_text),
            'max_text': self.max_text,
            'max_text_length': len(self.max_text),
            'min_rows': self.min_rows,
            'max_rows': self.max_rows,
            'min_cols': self.min_cols,
            'max_cols': self.max_cols,
        })

    def __str__(self):
        """
        Returns a string representation of the Dataset object with useful information.

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
            Lazy Mode               {self.lazy_mode}
            Seed                    {self.seed}

            \nDataset Information\n
            Total Size              {self.size}
            Reference Pixels        {self.reference_pixels}

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

    def batch_generator(self,
                        partition,
                        batch_size=16,
                        augmentor=None,
                        standardize=True,
                        shuffle=True,
                        debug=False):
        """
        Generates a batch of data samples for the specified partition.

        Parameters
        ----------
        partition : str
            The partition type ('training', 'validation', or 'test').
        batch_size : int, optional
            The number of samples in each batch, by default 16.
        augmentor : Augmentor, optional
            The Augmentor class. Default is None.
        standardize : bool, optional
            Indicates whether to standardize the batch, default is True.
        shuffle : bool, optional
            Specifies whether shuffles per epoch, default is True.
        debug : bool, optional
            Specifies whether to enable debug mode, default is False.

        Returns
        -------
        tuple
            A tuple containing the input data and corresponding labels.
        """

        dataset = getattr(self, partition)
        indices = np.arange(dataset['size'])

        batch_index = 0
        label_index = 2 if debug else 3

        while True:
            if batch_index >= dataset['size']:
                if shuffle:
                    np.random.shuffle(indices)
                batch_index = 0

            batch_indices = indices[batch_index:batch_index + batch_size]
            batch_index += batch_size

            x_data = []
            y_data = []

            if self.lazy_mode:
                batch_data = [dataset['data'][i] for i in batch_indices]

                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(self.read_image, data[0], data[1]) for data in batch_data]

                    for future, data in zip(futures, batch_data):
                        x_data.append(future.result())
                        y_data.append(data[label_index])
            else:
                for i in batch_indices:
                    x_data.append(dataset['data'][i][0])
                    y_data.append(dataset['data'][i][label_index])

            if augmentor:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    futures = [executor.submit(augmentor.augmentation, x, x_data) for x in x_data]
                    x_data = [future.result() for future in futures]

            if standardize:
                axis = np.array([x.shape for x in x_data])
                max_axis = [np.max(axis[..., 0]), np.max(axis[..., 1])]

                with concurrent.futures.ProcessPoolExecutor() as executor:
                    futures = [executor.submit(self.standardize_image, x, max_axis) for x in x_data]
                    x_data = [future.result() for future in futures]

                x_data = np.array(x_data, dtype=np.float64)
                y_data = np.array(y_data, dtype=np.int16)

            # batch = (x_data,) if 'test' in partition else (x_data, y_data)

            yield x_data, y_data

    def standardize_text(self, text):
        """
        Standardize texts by string formatting.

        Parameters
        ----------
        texts : list
            The texts to be standardized.

        Returns
        -------
        list
            The standardized texts.
        """

        if isinstance(text, str):
            text = [text.split('\n')]

        if isinstance(text[0], str):
            text = [text]

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

        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        detokenizer = nltk.tokenize.TreebankWordDetokenizer()

        for i in range(len(text)):
            text[i] = [line.strip() for line in text[i] if line.strip()]

            for j, line in enumerate(text[i]):
                line = html.unescape(line)

                for pattern, replacement in regexes.items():
                    line = pattern.sub(replacement, line)

                line = re.sub(r'\s+([!?,.;:])', r'\1', line)
                line = re.sub(r"\b([A-Za-z]+'[A-Za-z]+)\b", r'\1', line)
                line = re.sub(r'\s+(?=[\'"])', '', line)
                line = re.sub(r'(?<=[\'"]) +', '', line)

                tokens = tokenizer.tokenize(line)
                line = detokenizer.detokenize(tokens)

                line = re.sub(r'(.*?)"\s(.*?)\s"(.*?)', r'\1"\2"\3', line.replace('"', ' " ')).strip()
                line = line.translate(str.maketrans({punct: f" {punct} " for punct in string.punctuation}))
                line = re.sub(r'\s+', ' ', line.strip()).strip()

                text[i][j] = line

        return text

    def standardize_image(self, image, max_axis=None):
        """
        Standardize the given image for optical model input.

        Parameters
        ----------
        image : numpy.ndarray
            The input image to standardize.
        max_axis : tuple or None, optional
            The maximum axis size to pad the image.
            The format of max_axis should be (height, width).

        Returns
        -------
        numpy.ndarray
            The standardized image.
        """

        if max_axis is not None:
            bottom_pad = max(0, max_axis[0] - image.shape[0])
            right_pad = max(0, max_axis[1] - image.shape[1])

            image = cv2.copyMakeBorder(image, 0, bottom_pad, 0, right_pad,
                                       borderType=cv2.BORDER_CONSTANT,
                                       value=self.reference_pixels[0])

        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image, 1)

        image = np.divide(image, 255, dtype=np.float32)

        return image

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

    def _prepare_data(self, data, infer=False):
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

        reference_pixels = []

        for i in range(len(data)):
            if not data[i]:
                continue

            np.random.shuffle(data[i])

            with multiprocessing.get_context('fork').Pool() as pool:
                data[i] = [list(x) for x in pool.map(self._validate_data_item, data[i]) if x]

            reference_pixels.extend([sublist[-1] for sublist in data[i]])
            data[i] = [sublist[:-1] for sublist in data[i]]

        reference_pixels = np.average(np.transpose(reference_pixels), axis=1)
        reference_pixels = reference_pixels.astype(np.uint8).tolist()

        return data, reference_pixels

    def _validate_data_item(self, item, infer=False):
        """
        Validates a data item.

        Parameters
        ----------
        item : tuple
            The data item to validate.
        infer : bool, optional
            Flag for inference mode. Default is False.

        Returns
        -------
        tuple
            The validated data item.
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

        label = self.standardize_text(label)[0]

        if not infer and not label:
            print(f"Image `{os.path.basename(image_path)}` has an invalid label.")
            return None

        reference_pixels = [np.min(image), np.average(image), np.max(image)]
        image = image_path if self.lazy_mode else image

        return image, bbox, label, reference_pixels

    def _create_partition_dictionary(self, partition_data, test=False):
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
            'data': partition_data,
            'size': len(partition_data),
            'corpus': '',
            'charset': [],
            'min_text': '',
            'max_text': '',
            'min_rows': 0,
            'max_rows': 0,
            'min_cols': 0,
            'max_cols': 0,
        }

        labels = [x[2] for x in partition_data if x[2]]

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
        self.shape = (max_rows, max_cols + (len(self.charset) - len(charset)))

    def encode_data(self, data):
        """
        Encode the data by mapping labels to their corresponding token indices.

        Parameters
        ----------
        data : list
            List of data to encode, where each element is [image, bbox, label].

        Returns
        -------
        list
            Encoded data with token indices appended to each element.
        """

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.encode, x[-1]) for x in data]
            encoded_labels = [future.result() for future in futures]

            for i in range(len(data)):
                data[i].append(encoded_labels[i])

        return data

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

        pad_tk_index = self.charset.index(self.pad_tk)
        unk_tk_index = self.charset.index(self.unk_tk)
        sos_tk_index = self.charset.index(self.sos_tk)
        eos_tk_index = self.charset.index(self.eos_tk)

        encoded_label = []

        for row in label:
            enconded_row = [sos_tk_index]

            for char in row:
                index = self.charset.index(char) if char in self.charset else unk_tk_index
                enconded_row.append(index)

            enconded_row += [eos_tk_index]
            enconded_row += [pad_tk_index] * max(0, abs(self.shape[1] - len(enconded_row)))

            encoded_label.append(enconded_row)

        return encoded_label

    def decode_data(self, data):
        """
        Decode the data by converting token indices back to their corresponding characters.

        Parameters
        ----------
        data : list
            List of data to decode, where each element is [image, bbox, label].

        Returns
        -------
        list
            Decoded data with labels appended to each element.
        """

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self.decode, x[-1]) for x in data]
            labels = [future.result() for future in futures]

            for i in range(len(data)):
                data[i].append(labels[i])

        return data

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
            row = row.replace(self.sos_tk, '')
            row = row.replace(self.eos_tk, '')

            label.append(row)

        return label
