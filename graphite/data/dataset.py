import os
import random
import zipfile
import importlib
import numpy as np
import concurrent.futures

from data import utils
from data.tokenizer import Tokenizer


class Dataset():
    """
    General data source management.
    """

    def __init__(self,
                 mode='recognition',
                 source=None,
                 text_level=None,
                 image_shape=None,
                 training_ratio=None,
                 validation_ratio=None,
                 test_ratio=None,
                 lazy_mode=False,
                 data=None,
                 tokenizer=Tokenizer(),
                 artifact_path='dataset',
                 seed=None):
        """
        Initialize the Dataset instance.

        Parameters
        ----------
        mode : str, optional
            Dataset application mode.
        source : str, optional
            The data source name.
        text_level : str, optional
            The text structure level.
        image_shape : list, optional
            The images shape.
        training_ratio : float or int, optional
            The training ratio for resample.
        validation_ratio : float or int, optional
            The validation ratio for resample.
        test_ratio : float or int, optional
            The test ratio for resample.
        lazy_mode : bool, optional
            Enable lazy loading mode.
        data : list, optional
            Data for inference mode.
        tokenizer : object, optional
            Tokenizer used in input data.
        artifact_path : str, optional
            Path name to fetch the data.
        seed : int, optional
            Seed for random shuffle.
        """

        random.seed(seed)
        np.random.seed(seed)

        self.mode = mode
        self.source = source
        self.text_level = text_level
        self.image_shape = tuple(image_shape)
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.lazy_mode = lazy_mode
        self.tokenizer = tokenizer
        self.artifact_path = artifact_path
        self.seed = seed

        if data is None:
            self._source = self._import_source(self.source)
            self._source = self._source(self.artifact_path)

            if hasattr(self._source, 'base_path'):
                self._extract_source_zip(self.artifact_path, self._source.base_path)

            data = self._source.fetch_data(self.text_level)

        data = self._partitioning(data)
        data = self._validation(data)

        self.samples = self._build_samples(data)
        self.multigrams = self._build_multigrams(data)

    def __repr__(self):
        """
        Provides a formatted string with useful information.

        Returns
        -------
        str
            Formatted string with useful information.
        """

        info = '-------------------------------------------------'
        info += f'\n{self.__class__.__name__}'
        info += '\n-------------------------------------------------'
        info += f"\n{'Mode':<{26}}: {self.mode or '-'}"
        info += f"\n{'Source':<{26}}: {self.source or '-'}"
        info += f"\n{'Text Level':<{26}}: {self.text_level or '-'}"
        info += f"\n{'Image Shape':<{26}}: {self.image_shape or '-'}"
        info += f"\n{'Training Ratio':<{26}}: {self.training_ratio or '-'}"
        info += f"\n{'Validation Ratio':<{26}}: {self.validation_ratio or '-'}"
        info += f"\n{'Test Ratio':<{26}}: {self.test_ratio or '-'}"
        info += f"\n{'Lazy Mode':<{26}}: {self.lazy_mode}"
        info += f"\n{'Seed':<{26}}: {self.seed}"
        info += '\n-------------------------------------------------'
        info += f"\n{'Training Data':<{26}}: {len(self.samples['source']['training']):,}"
        info += f"\n{'Validaiton Data':<{26}}: {len(self.samples['source']['validation']):,}"
        info += f"\n{'Test Data':<{26}}: {len(self.samples['source']['test']):,}"
        info += f"\n{'Total Data':<{26}}: {sum(len(x) for x in self.samples['source'].values()):,}"
        info += '\n-------------------------------------------------'
        info += f"\n{'Multigrams':<{26}}: {len(self.multigrams['source']):,}"

        return info

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

    def _partitioning(self, data):
        """
        Prepares the data in partitioning.

        Parameters
        ----------
        data : dict
            The input data with the partitions as keys.

        Returns
        -------
        dict
            The partitioned data.
        """

        if not hasattr(self, '_source'):
            return data

        def parse_ratio(ratio):
            if ratio is None:
                return None
            if isinstance(ratio, str):
                return float(ratio) if '.' in ratio else int(ratio)
            return ratio

        data.setdefault('training', [])
        data.setdefault('validation', [])

        if len(data['training']) > 0 and len(data['validation']) == 0:
            np.random.shuffle(data['training'])
            index = max(round(0.1 * len(data['training'])), 1)

            data['validation'] = data['training'][:index]
            data['training'] = data['training'][index:]

        ratios = {
            'training': parse_ratio(self.training_ratio),
            'validation': parse_ratio(self.validation_ratio),
            'test': parse_ratio(self.test_ratio),
        }

        ratio = sum([ratios[i] for i in ratios if ratios[i] is not None])

        if ratio > 0:
            if isinstance(ratio, float) and ratio == 1.0:
                merged = []

                for i in ratios:
                    if ratios[i] is None:
                        continue

                    np.random.shuffle(data[i])
                    merged.extend(data[i])

                total_merged = len(merged)

                if total_merged > 0:
                    for i in ratios:
                        if ratios[i] is None:
                            continue

                        np.random.shuffle(merged)
                        index = round((ratios[i] + 1e-7) * total_merged)

                        data[i] = merged[:index]
                        merged = merged[index:]

            else:
                for i in ratios:
                    if ratios[i] is None:
                        continue

                    np.random.shuffle(data[i])
                    index = round((ratios[i] + 1e-7) * len(data[i])) \
                        if isinstance(ratios[i], float) else ratios[i]

                    data[i] = data[i][:index]

        return data

    def _validation(self, data):
        """
        Validates and cleans the data by removing invalid entries.

        Parameters
        ----------
        data : dict
            The input data to be validated, organized in partitions.

        Returns
        -------
        dict
            The cleaned data with invalid entries removed.
        """

        def validate(index, item):
            item['text'] = utils.format_text(item['text'])

            if not item['text'] and hasattr(self, '_source') is not None:
                print(f"Image `{os.path.basename(item['image'])}` has an invalid label.")
                return index

            if item.get('image'):
                if not os.path.exists(item['image']):
                    print(f"Image `{os.path.basename(item['image'])}` does not exist.")
                    return index

                try:
                    image = utils.read_image(item['image'], item['bbox'], self.image_shape)

                    if image is None or image.size == 0:
                        print(f"Image `{os.path.basename(item['image'])}` has an invalid size.")
                        return index

                except Exception:
                    print(f"Image `{os.path.basename(item['image'])}` cannot be read.")
                    return index

            return None

        for partition in data:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(validate, i, x) for i, x in enumerate(data[partition])]
                indices = [x for x in [future.result() for future in futures] if x is not None]

            for index in indices:
                del data[partition][index]

        return data

    def _build_samples(self, data):
        """
        Builds the samples from data partitions.

        Parameters
        ----------
        data : dict
            Dictionary containing data partitions.

        Returns
        -------
        dict
            A dictionary with raw and processed data.
        """

        samples = {'source': {}, 'encode': {}}
        keepstats = hasattr(self, '_source')

        def build(x):
            source = x.copy()
            encode = x.copy()

            if not self.lazy_mode:
                encode['image'] = utils.read_image(x['image'], x['bbox'], self.image_shape)

            encode['text'] = self.tokenizer.encode_text(x['text'], keepstats=keepstats)
            encode['writer'] = self.tokenizer.encode_writer(x['writer'], keepstats=keepstats)

            return source, encode

        for partition in data:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(build, x) for x in data[partition]]
                results = [future.result() for future in futures]

            source, encode = zip(*results)
            samples['source'][partition] = np.array(source, dtype=object)
            samples['encode'][partition] = np.array(encode, dtype=object)

        return samples

    def _build_multigrams(self, data):
        """
        Builds multigrams from the data partitions.

        Parameters
        ----------
        data : dict
            Dictionary containing data partitions.

        Returns
        -------
        dict
            A dictionary with raw and processed multigrams.
        """

        multigrams = {'source': [], 'encode': []}

        if 'synthesis' not in self.mode:
            return multigrams

        def build(x):
            words = x['text'].replace('\n\n', ' ').replace('\n', ' ').split()
            multigrams = []

            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    multigram = ' '.join(words[i:j])
                    multigrams.append(multigram)

            return multigrams

        for partition in data:
            if 'test' in partition:
                continue

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(build, x) for x in data[partition]]
                source = [multigram for future in futures for multigram in future.result()]

            np.random.shuffle(source)

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.tokenizer.encode_text, x, False) for x in source]
                encode = [future.result() for future in futures]

            multigrams['source'].extend(source)
            multigrams['encode'].extend(encode)

        multigrams['source'] = np.array(multigrams['source'], dtype=object)
        multigrams['encode'] = np.array(multigrams['encode'], dtype=object)

        return multigrams

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
#             The number of samples in each batch.
#         augmentor : Augmentor, optional
#             The Augmentor class.
#         raw_data : bool, optional
#             Specifies whether to generate raw or processed data.
#         shuffle : bool, optional
#             Specifies whether shuffles per epoch.

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
