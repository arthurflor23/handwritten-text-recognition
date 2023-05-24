import random
import importlib


class Dataset():

    def __init__(self,
                 source,
                 level,
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

        data = self._fetch_data()

        self.training = {
            'index': 0,
            'labels': data[0][0],
            'images': data[0][1],
            'cropping': data[0][2],
        }

        self.validation = {
            'index': 0,
            'labels': data[1][0],
            'images': data[1][1],
            'cropping': data[1][2],
        }

        self.test = {
            'index': 0,
            'labels': data[2][0],
            'images': data[2][1],
            'cropping': data[2][2],
        }

        print(self.training)
        print(self.validation)
        print(self.test)

    def __repr__(self):
        return "temp 1"

    def _fetch_data(self):
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

        # Perform data validation checks
        assert data is not None and 1 <= len(data) <= 3, "data must have 3 dims (training, validation, test)"

        if isinstance(data, tuple):
            data = list(data)

        data.extend([[[], [], []]] * (3 - len(data)))

        for i in range(len(data)):
            data[i] = data[i] or [[], [], []]
            data[i].extend([[[], [], []]] * (3 - len(data[i])))
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
                    index = round(ratio * total_merged)
                    data[i] = merged[:index]
                    merged[:index] = []

        else:
            for i, ratio in enumerate(ratios):
                random.shuffle(data[i])

                if ratio is not None:
                    index = round(ratio * len(data[i])) if isinstance(ratio, float) else ratio
                    data[i] = data[i][:index]

        # Unzip the data and convert to lists
        data = [[list(x) for x in zip(*d)] if d else [[], [], []] for d in data]

        return data
