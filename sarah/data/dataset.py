import os
import zipfile
import importlib
import numpy as np
import concurrent.futures

from sarah.data import utils
from sarah.data.tokenizer import Tokenizer


class Dataset():
    """
    General data source management.
    """

    def __init__(self,
                 source=None,
                 text_level='line',
                 image_shape=(1024, 64, 1),
                 char_width=0,
                 mask_by_text=False,
                 order_by_text=False,
                 training_ratio=None,
                 validation_ratio=None,
                 test_ratio=None,
                 illumination=False,
                 binarization=None,
                 lazy_mode=False,
                 data=None,
                 tokenizer=None,
                 multigrams=False,
                 input_path='datasets',
                 seed=None):
        """
        Initialize the Dataset instance.

        Parameters
        ----------
        source : str, optional
            The data source name.
        text_level : str, optional
            The text structure level.
        image_shape : list, optional
            The images shape.
        char_width : int, optional
            The width per character.
        mask_by_text : bool, optional
            Whether to mask data by text length.
        order_by_text : bool, optional
            Whether to sort data by text length.
        training_ratio : float or int, optional
            The training ratio for resample.
        validation_ratio : float or int, optional
            The validation ratio for resample.
        test_ratio : float or int, optional
            The test ratio for resample.
        illumination : bool, optional
            Apply illumination compensation.
        binarization : str, optional
            Apply binarization method.
        lazy_mode : bool, optional
            Enable lazy loading mode.
        data : list, optional
            Data for inference mode.
        tokenizer : object, optional
            Tokenizer used in input data.
        multigrams : bool, optional
            Enable multigrams process.
        input_path : str, optional
            Path to input data.
        seed : int, optional
            Seed for random shuffle.
        """

        if seed is not None:
            np.random.seed(seed)

        self.source = source
        self.text_level = text_level
        self.image_shape = image_shape
        self.char_width = char_width
        self.mask_by_text = mask_by_text
        self.order_by_text = order_by_text
        self.training_ratio = training_ratio
        self.validation_ratio = validation_ratio
        self.test_ratio = test_ratio
        self.illumination = illumination
        self.binarization = binarization
        self.lazy_mode = lazy_mode
        self.tokenizer = tokenizer or Tokenizer()
        self.multigrams = multigrams
        self.input_path = input_path
        self.seed = seed

        if data is None:
            self._source = self._import_source(self.source)
            self._source = self._source(self.input_path)

            if hasattr(self._source, 'base_path'):
                self._extract_source_zip(self.input_path, self._source.base_path)

            data = self._source.fetch_data(self.text_level)

            for items in data.values():
                items.sort(key=lambda x: x.get('image'), reverse=False)

        data = self._partitioning(data)

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

        pad, width = 25, 68

        info = "=" * width
        info += f"\n{self.__class__.__name__.center(width)}"
        info += "\n" + "-" * width
        info += f"\n{'source':<{pad}}: {self.source or '-'}"
        info += f"\n{'text_level':<{pad}}: {self.text_level or '-'}"
        info += f"\n{'image_shape':<{pad}}: {self.image_shape or '-'}"
        info += f"\n{'char_width':<{pad}}: {self.char_width or '-'}"
        info += f"\n{'training_ratio':<{pad}}: {self.training_ratio or '-'}"
        info += f"\n{'validation_ratio':<{pad}}: {self.validation_ratio or '-'}"
        info += f"\n{'test_ratio':<{pad}}: {self.test_ratio or '-'}"
        info += f"\n{'illumination':<{pad}}: {self.illumination or '-'}"
        info += f"\n{'binarization':<{pad}}: {self.binarization or '-'}"
        info += f"\n{'lazy_mode':<{pad}}: {self.lazy_mode}"
        info += f"\n{'seed':<{pad}}: {self.seed}"
        info += "\n" + "-" * width
        info += f"\n{'training_data':<{pad}}: {len(self.samples['source']['training']):,}"
        info += f"\n{'validation_data':<{pad}}: {len(self.samples['source']['validation']):,}"
        info += f"\n{'test_data':<{pad}}: {len(self.samples['source']['test']):,}"
        info += f"\n{'total_data':<{pad}}: {sum(len(x) for x in self.samples['source'].values()):,}"
        info += "\n" + "-" * width
        info += f"\n{'multigrams':<{pad}}: {len(self.multigrams['source']):,}"
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

    def _extract_source_zip(self, input_path, source):
        """
        Extracts a .zip file into a directory if the directory doesn't exist yet.

        Parameters
        ----------
        input_path : str, optional
            Path to input data.
        source : str
            The base name of the .zip file and target directory.
        """

        if not source.startswith(input_path):
            source = os.path.join(input_path, source)

        if not os.path.exists(source) and os.path.isfile(f'{source}.zip'):
            with zipfile.ZipFile(f'{source}.zip', 'r') as zip_ref:
                zip_ref.extractall(input_path)

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

        def parse_ratio(ratio):
            if ratio is None:
                return None
            if isinstance(ratio, str):
                return float(ratio) if '.' in ratio else int(ratio)
            return ratio

        data.setdefault('training', [])
        data.setdefault('validation', [])

        ratios = {
            'training': parse_ratio(self.training_ratio),
            'validation': parse_ratio(self.validation_ratio),
            'test': parse_ratio(self.test_ratio),
        }

        ratio_list = [ratios[i] for i in ratios if ratios[i] is not None]

        if len(ratio_list) > 0:
            ratio = sum(ratio_list)

            if isinstance(ratio, float) and ratio == 1.0:
                merged = []

                for i in ratios:
                    if ratios[i] is None:
                        continue

                    merged.extend(data[i])

                total_merged = len(merged)

                if total_merged > 0:
                    np.random.shuffle(merged)

                    for i in ratios:
                        if ratios[i] is None:
                            continue

                        index = round(ratios[i] * total_merged)
                        data[i] = merged[:index]
                        merged = merged[index:]

            else:
                for i in ratios:
                    if ratios[i] is None:
                        continue

                    if ratios[i] > 0:
                        np.random.shuffle(data[i])

                    index = round(ratios[i] * len(data[i])) \
                        if isinstance(ratios[i], float) else ratios[i]

                    data[i] = data[i][:index]

        return data

    def _build_samples(self, data):
        """
        Validates and builds the samples from data partitions.

        Parameters
        ----------
        data : dict
            Dictionary containing data partitions.

        Returns
        -------
        dict
            A dictionary with source and encoded data.
        """

        samples = {'source': {}, 'encoded': {}}
        keepstats = hasattr(self, '_source')

        def validate_and_build(item):
            item['text'] = utils.format_text(item['text'] or '')

            if not item['text'] and hasattr(self, '_source'):
                print(f"Image `{item['image']}` has an invalid label.")
                return None

            if item.get('image', None):
                if not os.path.isfile(item['image']):
                    print(f"Image `{item['image']}` does not exist.")
                    return None

                try:
                    image = utils.resize_image(image=utils.read_image(item['image'], item['bbox']),
                                               target_width=len(item['text']) * self.char_width,
                                               target_shape=self.image_shape)

                    if image is None or image.size < 16:
                        invalid_size = f"{image.shape[0]}x{image.shape[1]}"
                        print(f"Image `{item['image']}` is smaller than valid size ({invalid_size}).")
                        return None

                except Exception:
                    print(f"Image `{item['image']}` cannot be read.")
                    return None
            else:
                image = np.ones(shape=self.image_shape[:-1]) * 255

            source = item.copy()
            encoded = item.copy()

            if not self.lazy_mode:
                encoded['image'] = image

            encoded['text'] = self.tokenizer.encode_text(item['text'], keepstats=keepstats)
            encoded['writer'] = self.tokenizer.encode_writer(item['writer'], keepstats=keepstats)

            return source, encoded

        for partition in data:
            samples['source'][partition] = []
            samples['encoded'][partition] = []

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(validate_and_build, item) for item in data[partition]]
                results = [future.result() for future in futures if future.result() is not None]

            if results:
                if self.order_by_text:
                    results.sort(key=lambda x: len(x[0]['text']), reverse=True)

                source, encoded = zip(*results)

                samples['source'][partition] = np.array(source, dtype=object)
                samples['encoded'][partition] = np.array(encoded, dtype=object)

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

        if not self.multigrams:
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

            source = multigrams.copy()
            encoded = [self.tokenizer.encode_text(x, keepstats=False) for x in multigrams]

            return source, encoded

        for partition in data:
            if 'test' in partition:
                continue

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(build, x) for x in data[partition]]
                results = [future.result() for future in futures if future.result() is not None]

            if results:
                flattened = [(s, e) for x in results for s, e in zip(x[0], x[1])]

                if self.order_by_text:
                    flattened.sort(key=lambda x: len(x[0]), reverse=True)

                source, encoded = zip(*flattened)

                for s, e in zip(source, encoded):
                    multigrams['source'].append({'text': s})
                    multigrams['encoded'].append({'text': e})

        multigrams['source'] = np.array(multigrams['source'], dtype=object)
        multigrams['encoded'] = np.array(multigrams['encoded'], dtype=object)

        return multigrams

    def get_generator(self,
                      data_partition,
                      batch_size=8,
                      batch_encoded=True,
                      batch_processing=True,
                      batch_scale=True,
                      augmentor=None,
                      samples=None,
                      shuffle=False):
        """
        Generates a batch of samples for the partition.

        Parameters
        ----------
        data_partition : str
            The dataset partition which will be create the generator.
        batch_size : int, optional
            The number of samples in each batch.
        batch_encoded : bool, optional
            Specifies whether to use source or encoded data.
        batch_processing : bool, optional
            Specifies whether to process batch data for model input.
        batch_scale : bool, optional
            Specifies whether to scale batch data.
        augmentor : Augmentor, optional
            The Augmentor instance.
        samples : int, optional
            Fetch a specific number of samples.
        shuffle : bool, optional
            Specifies whether data is shuffled by epoch.

        Returns
        -------
        tuple
            The generator of batches and the steps per epoch.
        """

        def batch_generator(data, multigrams):
            data_length = len(data)
            multigram_length = len(multigrams)

            indices = np.arange(data_length)
            batch_index = 0

            while True:
                if self.order_by_text:
                    if shuffle:
                        batch_index = np.random.randint(0, data_length - batch_size)
                    elif batch_index >= data_length:
                        batch_index = 0

                    batch = data[batch_index:batch_index + batch_size]

                else:
                    if batch_index >= data_length:
                        batch_index = 0
                    if shuffle and batch_index == 0:
                        np.random.shuffle(indices)

                    batch = data[indices[batch_index:batch_index + batch_size]]

                batch_index += batch_size

                image_data, text_data, writer_data, segmentation_data = map(
                    list, zip(*[(x['image'], x['text'], x['writer'], None) for x in batch]))

                mask_data = utils.batch_masking(text_data if self.mask_by_text else image_data)

                aug_text_data = None
                aug_mask_data = None
                aug_image_data = None
                aug_segmentation_data = None

                if batch_encoded:
                    writer_data = np.array(writer_data)

                    if self.lazy_mode:
                        image_data = [
                            utils.resize_image(image=utils.read_image(x['image'], x['bbox']),
                                               target_width=len(x['text']) * self.char_width,
                                               target_shape=self.image_shape)
                            for x in batch
                        ]

                        if not self.mask_by_text:
                            mask_data = utils.batch_masking(image_data)

                    segmentation_data = utils.batch_binarization(image_data, method='sauvola', invert=True)

                    aug_text_data = text_data.copy()
                    aug_mask_data = mask_data.copy()
                    aug_image_data = image_data.copy()
                    aug_segmentation_data = segmentation_data.copy()

                    if augmentor:
                        aug_image_data = [
                            utils.resize_image(image=augmentor.augmentation(x, aug_image_data),
                                               target_shape=self.image_shape)
                            for x in aug_image_data
                        ]

                        if not self.mask_by_text:
                            aug_mask_data = utils.batch_masking(aug_image_data)

                        aug_segmentation_data = utils.batch_binarization(aug_image_data, method='sauvola', invert=True)

                    if multigram_length:
                        g_index = np.random.randint(0, multigram_length - len(batch))
                        aug_text_data = [x['text'] for x in multigrams[g_index:g_index + len(batch)]]
                        aug_mask_data = utils.batch_masking(aug_text_data)

                    if batch_processing:
                        text_data = utils.batch_processing(batch_mode='text',
                                                           batch_data=text_data,
                                                           padding_shape=self.tokenizer.lexical_shape)

                        aug_text_data = utils.batch_processing(batch_mode='text',
                                                               batch_data=aug_text_data,
                                                               padding_shape=self.tokenizer.lexical_shape)

                        mask_data = utils.batch_processing(batch_mode='binary',
                                                           batch_data=mask_data,
                                                           batch_scale=batch_scale,
                                                           padding_shape=self.image_shape)

                        aug_mask_data = utils.batch_processing(batch_mode='binary',
                                                               batch_data=aug_mask_data,
                                                               batch_scale=batch_scale,
                                                               padding_shape=self.image_shape)

                        image_data = utils.batch_processing(batch_mode='image',
                                                            batch_data=image_data,
                                                            batch_scale=batch_scale,
                                                            padding_shape=self.image_shape,
                                                            illumination=self.illumination,
                                                            binarization=self.binarization)

                        aug_image_data = utils.batch_processing(batch_mode='image',
                                                                batch_data=aug_image_data,
                                                                batch_scale=batch_scale,
                                                                padding_shape=self.image_shape,
                                                                illumination=self.illumination,
                                                                binarization=self.binarization)

                        segmentation_data = utils.batch_processing(batch_mode='binary',
                                                                   batch_data=segmentation_data,
                                                                   batch_scale=batch_scale,
                                                                   padding_shape=self.image_shape)

                        aug_segmentation_data = utils.batch_processing(batch_mode='binary',
                                                                       batch_data=aug_segmentation_data,
                                                                       batch_scale=batch_scale,
                                                                       padding_shape=self.image_shape)

                x_data = (aug_image_data, aug_text_data, writer_data, aug_mask_data, aug_segmentation_data)
                y_data = (image_data, text_data, writer_data, mask_data, segmentation_data)

                yield x_data, y_data

        subset = 'encoded' if batch_encoded else 'source'

        if samples is None:
            data = self.samples[subset][data_partition]
        else:
            q1_len = (samples // 4) + (1 if (samples % 4) > 0 else 0)
            q2_len = (samples // 4) + (1 if (samples % 4) > 1 else 0)
            q3_len = (samples // 4) + (1 if (samples % 4) > 2 else 0)
            q4_len = (samples // 4)

            data_len = len(self.samples[subset][data_partition])

            q1_start = 0
            q2_start = data_len // 4
            q3_start = data_len // 2
            q4_start = (3 * data_len) // 4

            q1_samples = self.samples[subset][data_partition][q1_start:q1_start + q1_len]
            q2_samples = self.samples[subset][data_partition][q2_start:q2_start + q2_len]
            q3_samples = self.samples[subset][data_partition][q3_start:q3_start + q3_len]
            q4_samples = self.samples[subset][data_partition][q4_start:q4_start + q4_len]

            data = list(q1_samples) + list(q2_samples) + list(q3_samples) + list(q4_samples)
            data = np.array(data)

        multigrams = self.multigrams[subset]
        batch_size = min(len(data), batch_size)

        steps = int(np.ceil(len(data) / batch_size)) if batch_size else None
        generator = batch_generator(data, multigrams) if steps else None

        return generator, steps
