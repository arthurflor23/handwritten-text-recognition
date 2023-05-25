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
    Dataset class representing a general class for data source management.
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

        Args:
            data (list, optional): custom data for inference mode. Defaults to None.
            source (str, optional): The data source name. Defaults to None.
            level (str, optional): The recoginition level. Defaults to None.
            training_ratio (float or int, optional): The training ratio for resample. Defaults to None.
            validation_ratio (float or int, optional): The validation ratio for resample. Defaults to None.
            test_ratio (float or int, optional): The test ratio for resample. Defaults to None.
            data_path (str, optional): Path name to fetch the data. Defaults to 'data'.
            lazy_mode (bool, optional): Lazy mode flag for lazy loading process. Defaults to True.
            seed (int, optional): The random seed. Defaults to 42.
        """

        self.source = source
        self.level = level
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.base_path = os.path.join(os.path.dirname(__file__), '..', '..')
        self.data_path = os.path.join(self.base_path, data_path)
        self.lazy_mode = lazy_mode
        self.seed = seed

        self.size = 0
        self.charset = []
        self.min_label = ''
        self.max_label = ''

        # Load data at startup
        if data is None and self.source is not None:
            data = self._fetch_data_from_source()

        if data is not None:
            # Prepare data
            data = self._prepare_data(data)

            # Create partitions
            self.training = self._create_partition_dict(data[0])
            self.validation = self._create_partition_dict(data[1])
            self.test = self._create_partition_dict(data[2])

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the Dataset object.

        Returns:
            str: A JSON-formatted string containing the object's attributes.
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
            'charset': self.charset,
            'charset_length': len(self.charset),
            'min_label': self.min_label,
            'min_label_length': len(self.min_label),
            'max_label': self.max_label,
            'max_label_length': len(self.max_label),
        })

    def __str__(self):
        """
        Returns a string representation of the Dataset object with useful information.

        Returns:
            str: The string representation of the object.
        """

        info = f"""
            Dataset Configuration
            Source                    {self.source or '-'}
            Level                     {self.level or '-'}
            Training Ratio            {self.training_ratio or '-'}
            Validation Ratio          {self.validation_ratio or '-'}
            Test Ratio                {self.test_ratio or '-'}
            Lazy Mode                 {self.lazy_mode}
            Seed                      {self.seed}

            Dataset Information
            Total Size                {self.size}
            Charset                   {''.join(self.charset)}
            Charset Length            {len(self.charset)}
            Smallest Label            {self.min_label}
            Smallest Label Length     {len(self.min_label)}
            Biggest Label             {self.max_label}
            Biggest Label Length      {len(self.max_label)}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def _create_partition_dict(self, partition_data):
        """
        Creates a partition dictionary from the given partition data.

        Args:
            partition_data (tuple): The partition data containing labels, images, and cropping information.

        Returns:
            dict: The partition dictionary.
        """

        # Default particion dict structure
        images, cropping, labels = partition_data

        partition_dict = {
            'index': 0,
            'labels': labels,
            'images': images,
            'cropping': cropping,
            'size': len(labels),
            'charset': sorted(set(''.join(''.join(x) for x in labels))) if labels else [],
            'min_label': min([' '.join(sublist) for sublist in labels], key=len) if labels else '',
            'max_label': max([' '.join(sublist) for sublist in labels], key=len) if labels else '',
        }

        self.size += partition_dict['size']
        self.charset = sorted(set(self.charset + partition_dict['charset']))

        if partition_dict['min_label']:
            self.min_label = partition_dict['min_label'] if not self.min_label else \
                min(self.min_label, partition_dict['min_label'], key=len)

        if partition_dict['max_label']:
            self.max_label = partition_dict['max_label'] if not self.max_label else \
                max(self.max_label, partition_dict['max_label'], key=len)

        return partition_dict

    def _fetch_data_from_source(self):
        """
        Fetches the data from the specified data source.

        Returns:
            list: The fetched data.
        """

        # Get the module based on the source
        module_name = importlib.util.resolve_name(f".source.{self.source}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "source must be created"

        module = importlib.import_module(module_name, __package__)

        # Get the method based on the level
        method_name = f"get_{self.level}_data"
        assert hasattr(module, method_name), f"`{method_name}` method must be created"

        method = getattr(module, method_name)

        # Call the method to get the data
        data = method(self.data_path)

        return data

    def _prepare_data(self, data):
        """
        Prepares the data for partitioning.

        Args:
            data (tuple): The input data (training, validation, test).

        Returns:
            list: The prepared data.
        """

        data = list(data)

        # Wrap data in a list if it's strings or a list of strings
        if isinstance(data[0], str) or isinstance(data[0][0], str):
            data = [[], [], data]

        for i in range(len(data)):
            data[i] = data[i] or []

            if not data[i]:
                continue

            # Wrap data in a list if it's strings
            if isinstance(data[i][0], str):
                data[i] = [data[i], [], []]

            if len(data[i]) == 2:
                if isinstance(data[i][1][0], str):
                    # Prepare data considering images and labels as input
                    data[i] = [data[i][0], [], data[i][1]]
                else:
                    # Prepare data considering images and cropping as input
                    data[i] = [data[i][0], data[i][1], []]

            # Replace empty lists with []
            data[i] = [data[i][y] or [] for y in range(len(data[i]))]

        # Extend data list with empty lists to ensure length of 3
        data.extend([[]] * (3 - len(data)))

        for i in range(len(data)):
            # Extend partition list with empty lists to ensure length of 3
            data[i].extend([[]] * (3 - len(data[i])))
            assert len(data[i]) == 3, "partitions must have 3 dims (images, cropping, labels)"

            # Prepare cropping and labels values
            if len(data[i][1]) == 0:
                data[i][1] = [[]] * len(data[i][0])

            elif len(data[i][1]) == 1:
                data[i][1] = [data[i][1][0]] * len(data[i][0])

            if len(data[i][2]) == 0:
                data[i][2] = [''] * len(data[i][0])

            assert len(data[i][0]) == len(data[i][1]) == len(data[i][2]), "dims must have the same length"

            # Check if the cropping is valid
            sum_crop = len([x for x in data[i][1] if len(x) > 0])
            sum_crop_dims = sum(len(x) for x in data[i][1])

            assert sum_crop == 0 or sum_crop_dims == (sum_crop * 4), "cropping must have 4 dimensions"

        # Set the random seed
        random.seed(self.seed)

        # Convert data to a list of tuples
        data = [list(zip(x[0], x[1], x[2])) for x in data]

        # Get the training, validation, and test ratios
        ratios = [self.training_ratio, self.validation_ratio, self.test_ratio]
        ratio_is_not_none = [ratio for ratio in ratios if ratio is not None]

        if ratio_is_not_none:
            for y in range(len(ratios)):
                if not ratios[y]:
                    continue

                ratios[y] = float(ratios[y]) if '.' in ratios[y] else int(ratios[y])

            # Calculate the total ratio
            ratio = sum(x for x in ratios if x is not None)

            # Resample data based on aspect ratio
            if isinstance(ratio, float) and ratio == 1.0:
                merged = []

                for y, ratio in enumerate(ratios):
                    if ratio is not None:
                        random.shuffle(data[y])
                        merged.extend(data[y])

                if merged:
                    total_merged = len(merged)

                    for y, ratio in enumerate(ratios):
                        if ratio is not None:
                            random.shuffle(merged)
                            index = round((ratio + 1e-8) * total_merged)
                            data[y] = merged[:index]
                            merged[:index] = []

            else:
                for y, ratio in enumerate(ratios):
                    if ratio is not None:
                        random.shuffle(data[y])
                        index = round((ratio + 1e-8) * len(data[y])) if isinstance(ratio, float) else ratio
                        data[y] = data[y][:index]

        # Filter valid data
        for y in range(len(data)):
            # Apply the process_data function to each item in self.data in parallel
            with multiprocessing.Pool() as pool:
                random.shuffle(data[y])
                valid_items = [x for x in pool.map(self._validate_data_item, data[y]) if x]

            # Unzip the data and convert to lists
            data[y] = list(map(list, zip(*valid_items))) if valid_items else [[], [], []]

        return data

    def _validate_data_item(self, item):
        """
        Validates a single data item.

        Args:
            item (tuple): The data item to validate.

        Returns:
            tuple: The validated data item.
        """

        image_path, cropping, label = item
        image = None

        # Check if the image exist and is readable
        if os.path.exists(image_path) and os.path.isfile(image_path):
            try:
                image = self._read_image(image_path, cropping)

            except Exception:
                print(f"Image `{os.path.basename(image_path)}` cannot be read.")
                return None
        else:
            print(f"Image `{os.path.basename(image_path)}` does not exist or is not a file.")
            return None

        if image.size == 0:
            print(f"Image `{os.path.basename(image_path)}` has an invalid size.")
            return None

        # Standardize label
        label = self._format_label(label)

        # Check lazy mode to determine whether to keep the image loaded
        image = image_path if self.lazy_mode else image

        return image, cropping, label

    def _read_image(self, image_path, cropping=None):
        """
        Read an image from the given file path and perform optional cropping.

        Args:
            image_path (str): The path to the image file.
            cropping (tuple, optional): The cropping coordinates (x, y, width, height). Defaults to None.

        Returns:
            numpy.ndarray: The loaded image as a NumPy array.
        """

        # Load image in grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if cropping is None or len(cropping) != 4:
            # Return the entire image
            return image

        # Extract cropping values
        x, y, width, height = cropping

        # Convert x, y, width, height to integers if they are floats
        if isinstance(x, float):
            x = int(x * image.shape[1])

        if isinstance(y, float):
            y = int(y * image.shape[0])

        if isinstance(width, float):
            width = int(width * image.shape[1])

        if isinstance(height, float):
            height = int(height * image.shape[0])

        # Crop image using pixel-based coordinates
        image = image[y:y+height, x:x+width]

        return image

    def _format_label(self, label):
        """
        Standardizes a label by formatting, normalizing, and standardizing the string of text.

        Args:
            label (str): The label to be standardized.

        Returns:
            str: The standardized label.
        """

        if isinstance(label, str):
            label = label.split('\n')

        # Dictionary of substitutions
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
        tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
        # Treebank detokenizer
        detokenizer = nltk.tokenize.treebank.TreebankWordDetokenizer()

        for i in range(len(label)):
            # Replace HTML entities
            label[i] = html.unescape(label[i])

            # Perform the substitutions using a loop
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
