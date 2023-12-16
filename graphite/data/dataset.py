import os
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
                 workflow='recognition',
                 source=None,
                 text_level='line',
                 image_shape=(1024, 128, 1),
                 training_ratio=None,
                 validation_ratio=None,
                 test_ratio=None,
                 lazy_mode=False,
                 data=None,
                 tokenizer=None,
                 artifact_path='datasets',
                 seed=None):
        """
        Initialize the Dataset instance.

        Parameters
        ----------
        workflow : str, optional
            Dataset workflow.
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

        np.random.seed(seed)

        self.workflow = workflow
        self.source = source
        self.text_level = text_level
        self.image_shape = tuple(image_shape or [])
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.lazy_mode = lazy_mode
        self.tokenizer = tokenizer or Tokenizer()
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

        info = "=================================================="
        info += f"\n{self.__class__.__name__.center(50)}"
        info += "\n--------------------------------------------------"
        info += f"\n{'workflow':<{25}}: {self.workflow or '-'}"
        info += f"\n{'source':<{25}}: {self.source or '-'}"
        info += f"\n{'text_level':<{25}}: {self.text_level or '-'}"
        info += f"\n{'image_shape':<{25}}: {self.image_shape or '-'}"
        info += f"\n{'training_ratio':<{25}}: {self.training_ratio or '-'}"
        info += f"\n{'validation_ratio':<{25}}: {self.validation_ratio or '-'}"
        info += f"\n{'test_ratio':<{25}}: {self.test_ratio or '-'}"
        info += f"\n{'lazy_mode':<{25}}: {self.lazy_mode}"
        info += f"\n{'seed':<{25}}: {self.seed}"
        info += "\n--------------------------------------------------"
        info += f"\n{'training_data':<{25}}: {len(self.samples['source']['training']):,}"
        info += f"\n{'validaiton_data':<{25}}: {len(self.samples['source']['validation']):,}"
        info += f"\n{'test_data':<{25}}: {len(self.samples['source']['test']):,}"
        info += f"\n{'total_data':<{25}}: {sum(len(x) for x in self.samples['source'].values()):,}"
        info += "\n--------------------------------------------------"
        info += f"\n{'multigrams':<{25}}: {len(self.multigrams['source']):,}"
        info += f"\n{self.tokenizer}"

        return info

    def _import_source(self, source):
        """
        Dynamically imports and returns a class from a specified source.

        Parameters
        ----------
        source : str
            The name of the source to be imported.

        Returns
        -------
        class
            The dynamically imported class.
        """

        module_name = importlib.util.resolve_name(f".source.{source}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, 'source file must be created'

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
                        index = round((ratios[i] + 1e-8) * total_merged)

                        data[i] = merged[:index]
                        merged = merged[index:]

            else:
                for i in ratios:
                    if ratios[i] is None:
                        continue

                    np.random.shuffle(data[i])
                    index = round((ratios[i] + 1e-8) * len(data[i])) \
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

        samples = {'source': {}, 'encoded': {}}
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
            if not data[partition]:
                samples['source'][partition] = []
                samples['encoded'][partition] = []
                continue

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(build, x) for x in data[partition]]
                results = [future.result() for future in futures]

            source, encode = zip(*results)
            samples['source'][partition] = np.array(source, dtype=object)
            samples['encoded'][partition] = np.array(encode, dtype=object)

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

        multigrams = {'source': [], 'encoded': []}

        if 'synthesis' not in self.workflow:
            return multigrams

        def build(x):
            lines = x['text'].split('\n')
            max_line_length = max(len(line) for line in lines)

            words = x['text'].replace('\n', ' ').split()
            multigrams = []

            for i in range(len(words)):
                for j in range(i + 1, len(words) + 1):
                    multigram = ''
                    for u in range(i, j):
                        last_line = multigram.split('\n')[-1]
                        next_line_length = len(last_line) + len(words[u])

                        br = '\n' if u < j and next_line_length > max_line_length else ' '
                        multigram += f"{br}{words[u]}"

                    multigrams.append(multigram.strip())

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
                encoded = [future.result() for future in futures]

            multigrams['source'].extend([{'text': x} for x in source])
            multigrams['encoded'].extend([{'text': x} for x in encoded])

        multigrams['source'] = np.array(multigrams['source'], dtype=object)
        multigrams['encoded'] = np.array(multigrams['encoded'], dtype=object)

        return multigrams

    def get_generator(self,
                      partition,
                      batch_size=8,
                      augmentor=None,
                      use_source=False,
                      prepare_batch=True,
                      shuffle=False):
        """
        Generates a batch of samples for the partition.

        Parameters
        ----------
        partition : str
            The dataset partition which will be create the generator.
        batch_size : int, optional
            The number of samples in each batch.
        augmentor : Augmentor, optional
            The Augmentor instance.
        use_source : bool, optional
            Specifies whether to generate source or encoded data.
        prepare_batch : bool, optional
            Specifies whether to prepare data for model input.
        shuffle : bool, optional
            Specifies whether data is shuffled by epoch.

        Returns
        -------
        tuple
            The generator of batches and the steps per epoch.
        """

        def generator(subset, partition):
            samples_length = len(self.samples[subset][partition])
            multigrams_length = len(self.multigrams[subset])

            indices = np.arange(samples_length)
            batch_index = 0

            while True:
                if batch_index >= samples_length:
                    if shuffle:
                        np.random.shuffle(indices)
                    batch_index = 0

                batch_indices = indices[batch_index:batch_index + batch_size]
                batch_index += batch_size

                batch = self.samples[subset][partition][batch_indices]

                x_data = [data['image'] for data in batch]
                y_data = [data['text'] for data in batch]

                if not use_source:
                    if self.lazy_mode:
                        x_data = [utils.read_image(data['image'], data['bbox'], self.image_shape) for data in batch]

                    x_aug_data = x_data.copy()
                    y_aug_data = []

                    if multigrams_length:
                        multigrams_indices = np.random.choice(multigrams_length, batch_size, replace=False)
                        y_aug_data = [data['text'] for data in self.multigrams[subset][multigrams_indices]]

                    if augmentor:
                        x_aug_data = [augmentor.augmentation(x, x_aug_data) for x in x_aug_data]
                        x_aug_data = [utils.resize_image(x, self.image_shape) for x in x_aug_data]

                    if prepare_batch:
                        x_data = utils.prepare_image_batch(x_data, self.image_shape)
                        y_data = utils.prepare_text_batch(y_data, self.tokenizer.lexical_shape)

                        x_aug_data = utils.prepare_image_batch(x_aug_data, self.image_shape)
                        y_aug_data = utils.prepare_text_batch(y_aug_data, self.tokenizer.lexical_shape)

                    w_data = np.array([data['writer'] for data in batch], dtype=np.int64)
                    x_data = (x_data, y_data, x_aug_data, y_aug_data, w_data)

                yield (x_data, y_data)

        subset = 'source' if use_source else 'encoded'
        samples_length = len(self.samples[subset][partition])

        steps_per_epoch = int(np.ceil(samples_length / batch_size)) or None
        batch_generator = generator(subset, partition) if steps_per_epoch else None

        return batch_generator, steps_per_epoch
