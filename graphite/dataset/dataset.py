import os
import cv2
import json
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
        self.min_label = None
        self.max_label = None

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
            Charset                   {self.charset}
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
        labels, images, cropping = partition_data

        partition_dict = {
            'index': 0,
            'labels': labels,
            'images': images,
            'cropping': cropping,
            'size': len(labels),
            'charset': sorted(set(''.join(labels))) if labels else [],
            'min_label': min(labels, key=len) if labels else '',
            'max_label': max(labels, key=len) if labels else '',
        }

        self.size += partition_dict['size']
        self.charset = sorted(set(self.charset + partition_dict['charset']))

        if partition_dict['min_label']:
            self.min_label = partition_dict['min_label'] if self.min_label is None else \
                min(self.min_label, partition_dict['min_label'], key=len)

        if partition_dict['max_label']:
            self.max_label = partition_dict['max_label'] if self.max_label is None else \
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

        # Perform data validation checks
        assert data is not None and 1 <= len(data) <= 3, "data must have 3 dims (training, validation, test)"

        data = list(data)
        data.extend([[[], [], []]] * (3 - len(data)))

        for i in range(len(data)):
            data[i] = data[i] or [[], [], []]
            data[i].extend([[]] * (3 - len(data[i])))
            assert 1 <= len(data[i]) <= 3, "partitions must have 3 dims (labels, images, cropping)"

            data[i][2] = data[i][2] or []

            if len(data[i][2]) == 0:
                data[i][2] = [[]] * len(data[i][1])

            elif len(data[i][2]) == 1:
                data[i][2] = data[i][2] * len(data[i][1])

            assert len(data[i][0]) == len(data[i][1]) == len(data[i][2]), "dims must have the same length"

            sum_crop = len([x for x in data[i][2] if len(x) > 0])
            sum_crop_dims = sum(len(x) for x in data[i][2])

            assert sum_crop == 0 or sum_crop_dims == (sum_crop * 4), "cropping must have 4 dimensions"

        # Get the training, validation, and test ratios
        ratios = [self.training_ratio, self.validation_ratio, self.test_ratio]

        for i in range(len(ratios)):
            if ratios[i] is not None:
                ratios[i] = float(ratios[i]) if '.' in ratios[i] else int(ratios[i])

        # Calculate the total ratio
        ratio = sum(x for x in ratios if x is not None)

        # Set the random seed
        random.seed(self.seed)

        # Convert data to a list of tuples
        data = [list(zip(x[0], x[1], x[2])) for x in data]

        # Resample data based on aspect ratio
        if isinstance(ratio, float) and ratio == 1.0:
            merged = []

            for i, ratio in enumerate(ratios):
                if ratio is not None:
                    merged.extend(data[i])

            if merged:
                random.shuffle(merged)
                total_merged = len(merged)

                for i, ratio in enumerate(ratios):
                    if ratio is not None:
                        index = round((ratio + 1e-8) * total_merged)
                        data[i] = merged[:index]
                        merged[:index] = []

        else:
            for i, ratio in enumerate(ratios):
                if ratio is not None:
                    random.shuffle(data[i])
                    index = round((ratio + 1e-8) * len(data[i])) if isinstance(ratio, float) else ratio
                    data[i] = data[i][:index]

        # Filter valid data
        for i in range(len(data)):
            # Apply the process_data function to each item in self.data in parallel
            with multiprocessing.get_context('fork').Pool() as pool:
                valid_items = [x for x in pool.map(self._validate_data_item, iterable=data[i]) if x]

            # Unzip the data and convert to lists
            data[i] = list(map(list, zip(*valid_items))) if valid_items else [[], [], []]

        return data

    def _validate_data_item(self, item):
        """
        Validates a single data item.

        Args:
            item (tuple): The data item to validate.

        Returns:
            tuple: The validated data item.
        """

        item = list(item)
        image = None

        # Check if the image exist and is readable
        if os.path.exists(item[1]) and os.path.isfile(item[1]):
            try:
                image = cv2.imread(item[1], cv2.IMREAD_GRAYSCALE)
            except Exception:
                pass

        if image is None:
            print(f"Image `{os.path.basename(item[1])}` cannot be read.")
            return None

        if not self.lazy_mode:
            item[1] = image

        # Standardize label
        item[0] = self._format_label(item[0])

        return item

    def _format_label(self, label):
        """
        Standardizes a label by formatting, normalizing, and standardizing the string of text.

        Args:
            label (str): The label to be standardized.

        Returns:
            str: The standardized label.
        """

        if not label:
            return label

        # Perform formatting, normalization, and standardization operations on the label
        # ...

        return label
