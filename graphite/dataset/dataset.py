import os
import cv2
import random
import importlib


class Dataset():

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
        if data is None:
            data = self._fetch_data_from_source()

        # Prepare data
        data = self._prepare_data(data)

        # Create partitions
        self.training = self._create_partition_dict(data[0])
        self.validation = self._create_partition_dict(data[1])
        self.test = self._create_partition_dict(data[2])

    def __repr__(self):

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
        # Get the module based on the source
        module_name = f"dataset.source.{self.source}"
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "source must be created"

        module = importlib.import_module(module_name)

        # Get the method based on the level
        method_name = f"get_{self.level}_data"
        assert hasattr(module, method_name), f"`{method_name}` method must be created"

        method = getattr(module, method_name)

        # Call the method to get the data
        data = method(self.data_path)

        return data

    def _prepare_data(self, data):
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
            valid_items = []

            for item in data[i]:
                item = list(item)
                image = None

                if os.path.exists(item[1]) and os.path.isfile(item[1]):
                    try:
                        image = cv2.imread(item[1], cv2.IMREAD_GRAYSCALE)
                    except Exception:
                        pass

                if image is None:
                    print(f"Image `{os.path.basename(item[1])}` cannot be read.")
                    continue

                if not self.lazy_mode:
                    item[1] = image

                valid_items.append(item)

            # Unzip the data and convert to lists
            data[i] = list(map(list, zip(*valid_items))) if valid_items else [[], [], []]

        return data
