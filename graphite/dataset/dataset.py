import os
import re
import cv2
import json
import html
import nltk
import random
import importlib
import multiprocessing


class Dataset():
    """
    General class for data source management
    """

    def __init__(self,
                 data=None,
                 source=None,
                 level=None,
                 training_ratio=None,
                 validation_ratio=None,
                 test_ratio=None,
                 data_path='data',
                 lazy_mode=True,
                 seed=42):
        """
        Initializes a new instance of the Dataset class.

        Parameters
        ----------
        data : list, optional
            Custom data for inference mode. Defaults to None.
        source : str, optional
            The data source name. Defaults to None.
        level : str, optional
            The recognition level. Defaults to None.
        training_ratio : float or int, optional
            The training ratio for resample. Defaults to None.
        validation_ratio : float or int, optional
            The validation ratio for resample. Defaults to None.
        test_ratio : float or int, optional
            The test ratio for resample. Defaults to None.
        data_path : str, optional
            Path name to fetch the data. Defaults to 'data'.
        lazy_mode : bool, optional
            Lazy mode flag for lazy loading process. Defaults to True.
        seed : int, optional
            The random seed. Defaults to 42.

        Returns
        -------
        None
        """

        self.source = source
        self.level = level
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.base_path = os.path.join(os.path.dirname(__file__), '..', '..')
        self.data_path = os.path.join(self.base_path, data_path)
        self.lazy_mode = lazy_mode
        self.inference_mode = data is not None
        self.seed = seed

        self.size = 0
        self.charset = []

        self.min_text = ''
        self.max_text = ''

        self.min_rows = float('inf')
        self.max_rows = float('-inf')

        self.min_columns = float('inf')
        self.max_columns = float('-inf')

        # Load data at startup
        if not self.inference_mode:
            data = self._fetch_data_from_source()

        # Prepare data
        data = self._prepare_data(data)

        # Create partitions
        self.training = self._create_partition_dictionary(data[0])
        self.validation = self._create_partition_dictionary(data[1])
        self.test = self._create_partition_dictionary(data[2])

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
            'inference_mode': self.inference_mode,
            'seed': self.seed,
            'size': self.size,
            'charset': self.charset,
            'charset_length': len(self.charset),
            'min_text': self.min_text,
            'min_text_length': len(self.min_text),
            'max_text': self.max_text,
            'max_text_length': len(self.max_text),
            'min_rows': self.min_rows,
            'max_rows': self.max_rows,
            'min_columns': self.min_columns,
            'max_columns': self.max_columns,
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
            Dataset Configuration
            Source                  {self.source or '-'}
            Recognition Level       {self.level or '-'}
            Training Ratio          {self.training_ratio or '-'}
            Validation Ratio        {self.validation_ratio or '-'}
            Test Ratio              {self.test_ratio or '-'}
            Lazy Mode               {self.lazy_mode}
            Inference Mode          {self.inference_mode}
            Seed                    {self.seed}

            Dataset Information
            Total Size              {self.size}

            Training Size           {self.training['size']}
            Validation Size         {self.validation['size']}
            Test Size               {self.test['size']}

            Charset Length          {len(self.charset)}
            Charset                 {''.join(self.charset)}

            Min Text Length         {len(self.min_text)}
            Min Text                {self.min_text}

            Max Text Length         {len(self.max_text)}
            Max Text                {self.max_text}

            Min Rows                {self.min_rows}
            Max Rows                {self.max_rows}

            Min Columns             {self.min_columns}
            Max Columns             {self.max_columns}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def _create_partition_dictionary(self, partition_data):
        """
        Creates a partition dictionary from the given partition data.

        Parameters
        ----------
        partition_data : tuple
            The partition data containing labels, images, and bbox information.

        Returns
        -------
        dict
            The partition dict.
        """

        # Initialize the partition dictionary with default values
        dct = {
            'index': 0,
            'data': partition_data,
            'size': len(partition_data),
            'charset': [],
            'min_text': '',
            'max_text': '',
            'min_rows': 0,
            'max_rows': 0,
            'min_columns': 0,
            'max_columns': 0,
        }

        # Get the labels from the partition data
        labels = [x[2] for x in partition_data if x[2]]

        # Update the partition dictionary with relevant values if labels exist
        if labels:
            dct['charset'] = sorted(set(''.join(''.join(x) for x in labels)))
            dct['min_text'] = min(['\\n'.join(x) for x in labels], key=len)
            dct['max_text'] = max(['\\n'.join(x) for x in labels], key=len)
            dct['min_rows'] = min(len(x) for x in labels)
            dct['max_rows'] = max(len(x) for x in labels)
            dct['min_columns'] = min(len(y) for x in labels for y in x)
            dct['max_columns'] = max(len(y) for x in labels for y in x)

        # Update the object's properties using the partition dictionary
        self.size += dct['size']
        self.charset = sorted(set(self.charset + dct['charset']))

        if dct['min_text']:
            # Update the minimum label if it exists
            self.min_text = min(self.min_text or dct['min_text'], dct['min_text'], key=len)

        if dct['max_text']:
            # Update the maximum label if it exists
            self.max_text = max(self.max_text or dct['max_text'], dct['max_text'], key=len)

        if dct['min_rows']:
            # Update the minimum number of rows if it exists
            self.min_rows = min(self.min_rows, dct['min_rows'])

        if dct['max_rows']:
            # Update the maximum number of rows if it exists
            self.max_rows = max(self.max_rows, dct['max_rows'])

        if dct['min_columns']:
            # Update the minimum number of columns if it exists
            self.min_columns = min(self.min_columns, dct['min_columns'])

        if dct['max_columns']:
            # Update the maximum number of columns if it exists
            self.max_columns = max(self.max_columns, dct['max_columns'])

        return dct

    def _fetch_data_from_source(self):
        """
        Fetches the data from the specified data source.

        Returns
        -------
        list
            The fetched data.
        """

        # Check the module based on the source
        module_name = importlib.util.resolve_name(f".source.{self.source}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "source file must be created"

        # Import the module
        module = importlib.import_module(module_name, __package__)

        # Check the source class
        class_name = 'Source'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        # Create an instance of the class
        source = getattr(module, class_name)(self.data_path)

        # Check the method based on the level
        method_name = f"get_{self.level}_data"
        assert hasattr(source, method_name), f"`{method_name}` method must be created"

        # Get the method
        method = getattr(source, method_name)

        # Call the method to get the data
        data = method()

        return data

    def _prepare_data(self, data):
        """
        Prepares the data for partitioning.

        Parameters
        ----------
        data : tuple
            The input data (training, validation, test).

        Returns
        -------
        list
            The prepared data.
        """

        data = list(data)

        if not 1 <= len(data) <= 3:
            raise ValueError("input data must have 1 to 3 partition lists")

        if self.inference_mode:
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
                        # Prepare data considering images and labels as input
                        data[i][j] = [data[i][j][0] or '', [], data[i][j][1] or '']
                    else:
                        # Prepare data considering images and bbox as input
                        data[i][j] = [data[i][j][0] or '', data[i][j][1] or [], ['']]

                elif len(data[i][j]) == 3:
                    # Prepare data considering images, bbox, and labels as input
                    data[i][j] = [data[i][j][0] or '', data[i][j][1] or [], data[i][j][2] or '']

                if len(data[i][j][1]) != 0 and len(data[i][j][1]) != 4:
                    raise ValueError("bbox value must have 0 or 4 dims [x, y, width, height]")

                # Extend data list with empty lists to ensure length of 3
                data[i][j].extend([[]] * (3 - len(data[i][j])))

                # Extend data list with empty lists to ensure length of 3
            data[i].extend([[]] * (3 - len(data[i])))

        # Extend data list with empty lists to ensure length of 3
        data.extend([[]] * (3 - len(data)))

        # Set the random seed
        random.seed(self.seed)

        # Get the training, validation, and test ratios
        ratios = [self.training_ratio, self.validation_ratio, self.test_ratio]
        ratio_is_not_none = [ratio for ratio in ratios if ratio is not None]

        if ratio_is_not_none:
            for i in range(len(ratios)):
                if not ratios[i]:
                    continue

                if isinstance(ratios[i], str):
                    # Convert the ratio to a float or int
                    ratios[i] = float(ratios[i]) if '.' in ratios[i] else int(ratios[i])

            # Calculate the total ratio
            ratio = sum(x for x in ratios if x is not None)

            # Resample data based on aspect ratio
            if isinstance(ratio, float) and ratio == 1.0:
                merged = []

                for i, ratio in enumerate(ratios):
                    if ratio is not None:
                        random.shuffle(data[i])
                        merged.extend(data[i])

                if merged:
                    total_merged = len(merged)

                    for i, ratio in enumerate(ratios):
                        if ratio is not None:
                            random.shuffle(merged)
                            index = round((ratio + 1e-8) * total_merged)
                            data[i] = merged[:index]
                            merged = merged[index:]

            else:
                for i, ratio in enumerate(ratios):
                    if ratio is not None:
                        random.shuffle(data[i])
                        index = round((ratio + 1e-8) * len(data[i])) if isinstance(ratio, float) else ratio
                        data[i] = data[i][:index]

        # Filter valid data
        for i in range(len(data)):
            if not data[i]:
                continue

            # Apply the process_data function to items in parallel
            with multiprocessing.get_context('fork').Pool() as pool:
                random.shuffle(data[i])
                data[i] = [list(x) for x in pool.map(self._validate_data_item, data[i]) if x]

        return data

    def _validate_data_item(self, item):
        """
        Validates a data item.

        Parameters
        ----------
        item : tuple
            The data item to validate.

        Returns
        -------
        tuple
            The validated data item.
        """

        image_path, bbox, label = item
        image = None

        # Check if the image exist and is readable
        if os.path.exists(image_path) and os.path.isfile(image_path):
            try:
                image = self._read_image(image_path, bbox)

            except Exception:
                print(f"Image `{os.path.basename(image_path)}` cannot be read.")
                return None
        else:
            print(f"Image `{os.path.basename(image_path)}` does not exist or is not a file.")
            return None

        if image is None or image.size == 0:
            print(f"Image `{os.path.basename(image_path)}` has an invalid size.")
            return None

        # Standardize label
        label = self._format_label(label)

        if not self.inference_mode and not label:
            print(f"Image `{os.path.basename(image_path)}` has an invalid label.")
            return None

        # Check lazy mode to determine whether to keep the image loaded
        image = image_path if self.lazy_mode else image

        return image, bbox, label

    def _read_image(self, image_path, bbox=None):
        """
        Read an image from the given file path and perform optional bbox.

        Parameters
        ----------
        image_path : str
            The path to the image file.
        bbox : tuple, optional
            The bbox coordinates (x, y, width, height). Defaults to None.

        Returns
        -------
        numpy.ndarray
            The loaded image as a NumPy array.
        """

        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if bbox is None or len(bbox) != 4:
            # Return the entire image
            return image

        # Extract bbox values
        x, y, width, height = bbox

        # Convert x, y, width, height to integers if they are floats
        if isinstance(x, float):
            x = int(x * image.shape[1])

        if isinstance(y, float):
            y = int(y * image.shape[0])

        if isinstance(width, float):
            width = int(width * image.shape[1])

        if isinstance(height, float):
            height = int(height * image.shape[0])

        # Padding around the box
        y = max(0, abs(y - 10))
        x = max(0, abs(x - 10))
        height = min(image.shape[0], (height + 10))
        width = min(image.shape[1], (width + 10))

        # Crop image using pixel-based coordinates
        image = image[y:y+height, x:x+width]

        return image

    def _format_label(self, label):
        """
        Standardizes a label by formatting, normalizing, and standardizing the string of text.

        Parameters
        ----------
        label : str
            The label to be standardized.

        Returns
        -------
        str
            The standardized label.
        """

        if isinstance(label, str):
            label = label.split('\n')

        # Filter out empty strings
        label = [x.strip() for x in label if x.strip()]

        # Substitutions
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

        # Compile the regular expressions
        regexes = {re.compile(k): v for k, v in substitutions.items()}

        # Treebank tokenizer
        tokenizer = nltk.tokenize.TreebankWordTokenizer()
        # Treebank detokenizer
        detokenizer = nltk.tokenize.TreebankWordDetokenizer()

        for i in range(len(label)):
            # Replace HTML entities
            label[i] = html.unescape(label[i])

            # Perform the substitutions
            for pattern, replacement in regexes.items():
                label[i] = pattern.sub(replacement, label[i])

            # Remove extra spaces around punctuation marks
            label[i] = re.sub(r'\s+([!?,.;:])', r'\1', label[i])

            # Fix spacing around contractions
            label[i] = re.sub(r"\b([A-Za-z]+'[A-Za-z]+)\b", r'\1', label[i])

            # Remove spaces before opening single quotes
            label[i] = re.sub(r'\s+(?=[\'"])', '', label[i])

            # Remove spaces after closing single quotes
            label[i] = re.sub(r'(?<=[\'"]) +', '', label[i])

            # Tokenize text
            tokens = tokenizer.tokenize(label[i])
            # Detokenize the tokens
            label[i] = detokenizer.detokenize(tokens)

            # Remove extra spaces and handle quotes
            label[i] = re.sub(r'\s+', ' ', label[i].replace('"', ' " ')).strip()
            label[i] = re.sub(r'(.*?)"\s(.*?)\s"(.*?)', r'\1"\2"\3', label[i]).strip()

        return label
