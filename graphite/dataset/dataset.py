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
                 lazy_mode=True,
                 input_path=None,
                 seed=None):

        self.source = source
        self.level = level
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.lazy_mode = lazy_mode
        self.input_path = input_path
        self.seed = seed

        # Load the data upon initialization
        if data is None:
            data = self._fetch_data_from_source()

        # Validate data
        data = self._validate_data_from_source(data)
        # Setup data
        self._setup_data(data)

        print(self.training)
        print(self.validation)
        print(self.test)

    def __repr__(self):
        return "TEMP"

    def _setup_data(self, data):
        # Create a dataset dictionary with index, labels, images, and cropping
        def create_partition(data):
            return {
                'index': 0,
                'labels': data[0],
                'images': data[1],
                'cropping': data[2],
            }

        # Create the partitions
        self.training = create_partition(data[0])
        self.validation = create_partition(data[1])
        self.test = create_partition(data[2])

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
        data = method(self.input_path)

        return data

    def _validate_data_from_source(self, data):
        # Perform data validation checks
        assert data is not None and 1 <= len(data) <= 3, "data must have 3 dims (training, validation, test)"

        if isinstance(data, tuple):
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

        if isinstance(ratio, float) and ratio == 1.0:
            merged = []

            for i, ratio in enumerate(ratios):
                random.shuffle(data[i])

                if ratio is not None:
                    merged.extend(data[i])

            total_merged = len(merged)

            for i, ratio in enumerate(ratios):
                random.shuffle(data[i])

                if ratio is not None:
                    index = round((ratio + 1e-8) * total_merged)
                    data[i] = merged[:index]
                    merged[:index] = []

        else:
            for i, ratio in enumerate(ratios):
                random.shuffle(data[i])

                if ratio is not None:
                    index = round((ratio + 1e-8) * len(data[i])) \
                        if isinstance(ratio, float) else ratio
                    data[i] = data[i][:index]

        # Unzip the data and convert to lists
        data = [[list(x) for x in zip(*d)] if d else [[], [], []] for d in data]

        return data
