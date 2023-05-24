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

        self.training = {
            'index': 0,
            'labels': [],
            'images': [],
            'cropping': [],
        }

        self.validation = {
            'index': 0,
            'labels': [],
            'images': [],
            'cropping': [],
        }

        self.test = {
            'index': 0,
            'labels': [],
            'images': [],
            'cropping': [],
        }

        data = self._fetch_data()

        for x in data:
            print(x)

    def __repr__(self):
        return "temp 1"

    def _fetch_data(self):
        module_name = f"dataset.source.{self.source}"

        assert importlib.util.find_spec(module_name) is not None, "source must be created"
        module = importlib.import_module(module_name)

        method_name = f"get_{self.level}_data"

        assert hasattr(module, method_name), "`get_<level>_data` method must be created"
        method = getattr(module, method_name)

        data = method(self.input_path)
        data = self._validate_data(data)

        return data

    def _validate_data(self, data):
        assert data is not None, "no data"
        assert len(data) >= 1 and len(data) <= 3, "data in wrong format"

        if type(data) is tuple:
            data = list(data)

        data.extend([[[], [], []]] * (3 - len(data)))

        for i in range(len(data)):
            if data[i] is None:
                data[i] = [[], [], []]

            data[i].extend([[[], [], []]] * (3 - len(data[i])))
            assert len(data[i]) >= 1 and len(data[i]) <= 3, "data in wrong format"

            if data[i][2] is None:
                data[i][2] = []

            elif len(data[i][2]) == 0:
                data[i][2] = [[]] * len(data[i][1])

            elif len(data[i][2]) == 1:
                data[i][2] = data[i][2] * len(data[i][1])

            assert len(data[i][0]) == len(data[i][1])\
                and len(data[i][2]) == len(data[i][1]), "data lengths do not match"

        ratios = [self.training_ratio, self.validation_ratio, self.test_ratio]

        for i in range(len(ratios)):
            if ratios[i] is not None:
                ratios[i] = float(ratios[i]) if '.' in ratios[i] else int(ratios[i])

        ratio = sum(x for x in ratios if x is not None)

        random.seed(self.seed)
        data = [list(zip(x[0], x[1], x[2])) for x in data]

        if type(ratio) is float and ratio == 1.0:
            merged = []

            for i, ratio in enumerate(ratios):
                random.shuffle(data[i])

                if ratio is not None:
                    merged.extend(data[i])

            total_merged = len(merged)

            for i, ratio in enumerate(ratios):
                random.shuffle(data[i])

                if ratio is not None:
                    index = int(ratio * total_merged)
                    data[i] = merged[:index]
                    merged[:index] = []

        else:
            for i, ratio in enumerate(ratios):
                random.shuffle(data[i])

                if ratio is not None:
                    index = int(ratio * len(data[i])) if type(ratio) is float else ratio
                    data[i] = data[i][:index]

        for i in range(len(data)):
            unzipped = list(zip(*data[i]))
            data[i] = [list(x) for x in unzipped] if len(unzipped) else [[], [], []]

        return data
