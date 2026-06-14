import os
import glob
import zipfile
import importlib
import numpy as np
import concurrent.futures

from fontTools.ttLib import TTFont
from sarah.data import utils
from sarah.data.tokenizer import Tokenizer


class Dataset():
    """
    General data source management.
    """

    def __init__(self,
                 source=None,
                 text_level='line',
                 image_shape=(64, 1024, 1),
                 char_width=None,
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
                 fonts_path='fonts',
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
            The image dimensions.
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
            Tokenizer instance for input data.
        multigrams : bool, optional
            Enable multigrams process.
        input_path : str, optional
            Path to input data.
        fonts_path : str, optional
            Path to fonts data.
        seed : int, optional
            Seed for random shuffle.
        """

        if seed is not None:
            np.random.seed(seed)

        self.source = source
        self.text_level = text_level
        self.image_shape = image_shape
        self.char_width = char_width or 0
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
        self.fonts_path = fonts_path
        self.seed = seed

        if data is None:
            self._source = self._import_source_module(self.source)
            self._source = self._source(self.input_path)

            if hasattr(self._source, 'base_path'):
                self._extract_source_zip(self.input_path, self._source.base_path)

            data = self._source.fetch_data(self.text_level)

        data = self._load_partitions(data)
        self.samples = self._prepare_samples(data)
        self.multigrams = self._prepare_multigrams(data)

        fonts = self._load_fonts(self.fonts_path)
        self.font_graphs = self._prepare_graphs(fonts, self.tokenizer.chars, epsilon=0.01)

        self.tokenizer.set_font_graph_shape(self.font_graphs)

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
        info += "\n" + "-" * width
        info += f"\n{'fonts':<{pad}}: {len(self.font_graphs):,}"
        info += f"\n{self.tokenizer}"

        return info

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

    def _import_source_module(self, source):
        """
        Imports and returns a class from a specified source.

        Parameters
        ----------
        source : str
            The name of the source to be imported.

        Returns
        -------
        class
            The imported class module.
        """

        module_name = importlib.util.resolve_name(f".source.{source}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, 'source file must be created'

        module = importlib.import_module(module_name, __package__)

        class_name = 'Source'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        source = getattr(module, class_name)

        return source

    def _load_fonts(self, fonts_path):
        """
        Loads .ttf files from the given path that contain glyph outlines.

        Parameters
        ----------
        fonts_path : str
            Path to fonts folder.

        Returns
        -------
        list
            List of font entries with font and cmap.
        """

        fonts = []

        paths = glob.glob(os.path.join(fonts_path, '**', '*.ttf'), recursive=True)
        paths = list(set(paths))
        paths.sort()

        for path in paths:
            try:
                font = TTFont(path)
                cmap = font.getBestCmap() or {}

                if 'glyf' not in font:
                    continue

                fonts.append({'font': font, 'cmap': cmap})

            except Exception:
                print(f"Font `{path}` cannot be loaded.")

        return fonts

    def _load_partitions(self, data):
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

        ratios = {}

        for key in ['training', 'validation', 'test']:
            data.setdefault(key, [])

            data[key].sort(key=lambda x: x.get('text', ''), reverse=False)
            np.random.shuffle(data[key])

            ratio = getattr(self, f'{key}_ratio')

            if isinstance(ratio, str):
                ratios[key] = float(ratio) if '.' in ratio else int(ratio)
            else:
                ratios[key] = ratio

        ratio_list = [ratios[i] for i in ratios if ratios[i] is not None]

        if len(ratio_list) > 0:
            ratio = sum(ratio_list)

            if isinstance(ratio, float) and ratio == 1.0:
                merged = []

                for key in ratios:
                    if ratios[key] is None:
                        continue

                    merged.extend(data[key])

                total_merged = len(merged)

                if total_merged > 0:
                    for key in ratios:
                        if ratios[key] is None:
                            continue

                        index = round(ratios[key] * total_merged)
                        data[key] = merged[:index]
                        merged = merged[index:]

            else:
                for key in ratios:
                    if ratios[key] is None:
                        continue

                    index = round(ratios[key] * len(data[key])) \
                        if isinstance(ratios[key], float) else ratios[key]

                    data[key] = data[key][:index]

        return data

    def _prepare_graphs(self, fonts, chars, epsilon=0.0):
        """
        Builds graph representations from each character font.

        Parameters
        ----------
        fonts : str
            List of font entries with font and cmap.
        chars : list
            The tokenizer charset.
        epsilon : float, optional
            Maximum simplification applied to high-node outlier fonts.

        Returns
        -------
        list
            List of graph representations from each character font.
        """

        def build(font, eps):
            return [glyph_to_graph(font['font'], font['cmap'], x, eps) for x in chars]

        def compress_contour(points, eps):
            if len(points) < 3 or eps == 0:
                return np.ones(len(points), dtype=bool)

            a, b = points[0], points[-1]
            edge = b - a
            length = np.linalg.norm(edge)

            dists = np.linalg.norm(points - a, axis=1) if length == 0 else \
                np.abs(np.cross(edge, a - points)) / length

            peak = np.argmax(dists)
            mask = np.zeros(len(points), dtype=bool)

            if dists[peak] > eps:
                mask[:peak + 1] = compress_contour(points[:peak + 1], eps)
                mask[peak:] |= compress_contour(points[peak:], eps)
            else:
                mask[0] = mask[-1] = True

            return mask

        def count(graphs):
            return sum(len(graph['nodes']) for graph in graphs)

        def glyph_to_nodes(glyph):
            coords = np.array(glyph.coordinates)
            flags = np.uint8(glyph.flags)
            end_pts = set(glyph.endPtsOfContours)

            x_min, y_min = coords.min(axis=0)
            x_max, y_max = coords.max(axis=0)
            scale = max(x_max - x_min, y_max - y_min) or 1.0

            xy = (coords - [x_min, y_min]) / scale
            on_curve = (flags & 1).astype(np.float32)
            end_of_contour = np.array([i in end_pts for i in range(len(coords))])

            return np.stack([xy[:, 0], xy[:, 1], on_curve, end_of_contour], axis=1)

        def glyph_to_graph(font, cmap, char, eps):
            glyph_name = cmap.get(ord(char))

            if glyph_name is None:
                return {'nodes': [], 'edges': []}

            glyph = font['glyf'][glyph_name]

            if glyph.numberOfContours == 0:
                return {'nodes': [], 'edges': []}

            if glyph.numberOfContours < 0:
                components = [font['glyf'][x.glyphName] for x in glyph.components
                              if font['glyf'][x.glyphName].numberOfContours > 0]

                if not components:
                    return {'nodes': [], 'edges': []}

                glyphs = components
            else:
                glyphs = [glyph]

            pts, src, dst = [], [], []
            offset = 0

            for g in glyphs:
                nodes = glyph_to_nodes(g)
                bounds = [0] + [e + 1 for e in g.endPtsOfContours]
                contours = [np.arange(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]

                for idx in contours:
                    mask = compress_contour(nodes[idx, :2], eps)
                    kept = idx[mask]

                    if len(kept) < 2:
                        continue

                    pts.append(nodes[kept])

                    k = len(kept)
                    s = np.arange(k, dtype=np.int32) + offset
                    t = np.arange(1, k + 1, dtype=np.int32) % k + offset

                    src.append(np.concatenate([s, t]))
                    dst.append(np.concatenate([t, s]))

                    offset += k

            if not pts:
                return {'nodes': [], 'edges': []}

            return {
                'nodes': np.concatenate(pts, axis=0),
                'edges': np.stack([np.concatenate(src), np.concatenate(dst)], axis=0),
            }

        font_graphs = [build(font, 0.0) for font in fonts]

        if font_graphs and epsilon > 0:
            counts = np.array([count(graphs) for graphs in font_graphs])
            median = np.median(counts)

            q1, q3 = np.percentile(counts, [25, 75])
            threshold = q3 + 1.5 * (q3 - q1)

            for i in np.where(counts > threshold)[0]:
                best_graphs = font_graphs[i]
                best_count = counts[i]
                low, high = 0.0, epsilon

                for _ in range(10):
                    mid = (low + high) / 2.0
                    graphs = build(fonts[i], mid)
                    total = count(graphs)

                    if abs(total - median) < abs(best_count - median):
                        best_graphs = graphs
                        best_count = total

                    if total > median:
                        low = mid
                    else:
                        high = mid

                font_graphs[i] = best_graphs

        return font_graphs

    def _prepare_multigrams(self, data):
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

        def build(item):
            lines = item['text'].split('\n')
            max_line_length = max(len(line) for line in lines)

            words = item['text'].replace('\n', ' ').split()
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
                results = [x for x in (f.result() for f in futures) if x is not None]

            if not results:
                continue

            flattened = [(s, e) for x in results for s, e in zip(x[0], x[1])]

            if self.order_by_text:
                flattened.sort(key=lambda x: len(x[0]), reverse=False)

            source, encoded = zip(*flattened)

            for s, e in zip(source, encoded):
                multigrams['source'].append({'text': s})
                multigrams['encoded'].append({'text': e})

        multigrams['source'] = np.array(multigrams['source'], dtype=object)
        multigrams['encoded'] = np.array(multigrams['encoded'], dtype=object)

        return multigrams

    def _prepare_samples(self, data):
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

        def build(item):
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
                                               target_width=int(len(item['text']) * self.char_width),
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

            return source, encoded

        for partition in data:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(build, x) for x in data[partition]]
                results = [x for x in (f.result() for f in futures) if x is not None]

            if not results:
                samples['source'][partition] = np.array((), dtype=object)
                samples['encoded'][partition] = np.array((), dtype=object)
                continue

            if self.order_by_text:
                results.sort(key=lambda x: len(x[0]['text']), reverse=False)

            source, encoded = zip(*results)

            for item in encoded:
                item['text'] = self.tokenizer.encode_text(item['text'], keepstats=keepstats)
                item['writer'] = self.tokenizer.encode_writer(item['writer'], keepstats=keepstats)

            samples['source'][partition] = np.array(source, dtype=object)
            samples['encoded'][partition] = np.array(encoded, dtype=object)

        return samples

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

        def batch_processing_sync(data, aug_data, **kwargs):
            return (utils.batch_processing(batch_data=data, **kwargs),
                    utils.batch_processing(batch_data=aug_data, **kwargs))

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

                    batch_data = data[batch_index:batch_index + batch_size]

                else:
                    if batch_index >= data_length:
                        batch_index = 0

                    if shuffle and batch_index == 0:
                        np.random.shuffle(indices)

                    batch_data = data[indices[batch_index:batch_index + batch_size]]

                batch_index += batch_size
                batch_length = len(batch_data)

                image_data, text_data, writer_data = map(
                    list, zip(*[(x['image'], x['text'], x['writer']) for x in batch_data]))

                if batch_encoded and self.lazy_mode:
                    image_data = [
                        utils.resize_image(image=utils.read_image(x['image'], x['bbox']),
                                           target_width=int(len(x['text']) * self.char_width),
                                           target_shape=self.image_shape)
                        for x in batch_data
                    ]

                writer_data = np.array(writer_data)

                segmentation_data = utils.batch_binarization(batch_data=image_data,
                                                             method='sauvola',
                                                             invert=True)

                mask_data = utils.batch_masking(batch_data=text_data if self.mask_by_text else image_data,
                                                max_shape=self.image_shape,
                                                from_text=self.mask_by_text)

                font_graph_data = utils.batch_font_graph(font_graphs=self.font_graphs,
                                                         font_graph_shape=self.tokenizer.font_graph_shape,
                                                         batch_size=batch_length,
                                                         augmentor=False)

                aug_text_data = text_data.copy()
                aug_mask_data = mask_data.copy()
                aug_image_data = image_data.copy()
                aug_segmentation_data = segmentation_data.copy()
                aug_font_graph_data = [x.copy() for x in font_graph_data]

                if batch_encoded:
                    if augmentor:
                        aug_image_data = [
                            utils.resize_image(image=augmentor.augmentation(x, aug_image_data),
                                               target_shape=self.image_shape)
                            for x in aug_image_data
                        ]

                        aug_segmentation_data = utils.batch_binarization(batch_data=aug_image_data,
                                                                         method='sauvola',
                                                                         invert=True)

                        aug_font_graph_data = utils.batch_font_graph(font_graphs=self.font_graphs,
                                                                     font_graph_shape=self.tokenizer.font_graph_shape,
                                                                     batch_size=batch_length,
                                                                     augmentor=True)

                    if multigram_length:
                        index = np.random.randint(0, multigram_length - batch_length)
                        aug_text_data = [x['text'] for x in multigrams[index:index + batch_length]]

                        aug_mask_data = utils.batch_masking(batch_data=aug_text_data,
                                                            max_shape=self.image_shape,
                                                            from_text=True)
                    else:
                        aug_mask_data = aug_text_data if self.mask_by_text else aug_image_data
                        aug_mask_data = utils.batch_masking(batch_data=aug_mask_data,
                                                            max_shape=self.image_shape,
                                                            from_text=self.mask_by_text)

                    if batch_processing:
                        text_data, aug_text_data = batch_processing_sync(data=text_data,
                                                                         aug_data=aug_text_data,
                                                                         batch_mode='text',
                                                                         padding_shape=self.tokenizer.lexical_shape)

                        mask_data, aug_mask_data = batch_processing_sync(data=mask_data,
                                                                         aug_data=aug_mask_data,
                                                                         batch_mode='binary',
                                                                         batch_scale=batch_scale,
                                                                         padding_shape=self.image_shape)

                        image_data, aug_image_data = batch_processing_sync(data=image_data,
                                                                           aug_data=aug_image_data,
                                                                           batch_mode='image',
                                                                           batch_scale=batch_scale,
                                                                           padding_shape=self.image_shape,
                                                                           illumination=self.illumination,
                                                                           binarization=self.binarization)

                        segmentation_data, aug_segmentation_data = batch_processing_sync(data=segmentation_data,
                                                                                         aug_data=aug_segmentation_data,
                                                                                         batch_mode='binary',
                                                                                         batch_scale=batch_scale,
                                                                                         padding_shape=self.image_shape)

                x_data = [aug_image_data, aug_text_data, writer_data, aug_mask_data, aug_segmentation_data]
                y_data = [image_data, text_data, writer_data, mask_data, segmentation_data]

                if font_graph_data:
                    x_data.extend(aug_font_graph_data)
                    y_data.extend(font_graph_data)

                yield tuple(x_data), tuple(y_data)

        subset = 'encoded' if batch_encoded else 'source'

        if samples is None:
            data = self.samples[subset][data_partition]

        else:
            data_length = len(self.samples[subset][data_partition])

            sizes = [samples // 4 + (1 if i < samples % 4 else 0) for i in range(4)]
            offsets = [(i * data_length) // 4 for i in range(4)]

            data = np.array([
                sample
                for offset, size in zip(offsets, sizes)
                for sample in self.samples[subset][data_partition][offset:offset + size]
            ])

        multigrams = self.multigrams[subset]
        batch_size = min(len(data), batch_size)

        steps = int(np.ceil(len(data) / batch_size)) if batch_size else None
        generator = batch_generator(data, multigrams) if steps else None

        return generator, steps
