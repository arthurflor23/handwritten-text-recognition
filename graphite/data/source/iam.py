import os
import glob
import multiprocessing


class Source():
    """
    Represents the IAM database source.

    Requires implementation of `fetch_data`, returning a dictionary with
        'training', 'validation', and 'test' keys, each mapping to a list of data entries.

    Each data entry is a dictionary with keys 'image', 'bbox', 'text', and 'writer':
        'image' : str
            path to the image.
        'bbox' : list
            bounding box coordinates [x, y, h, w] (empty if no bbox).
        'text' : str
            text content with '\n' as line breaks.
        'writer' : str
            writer's unique ID ('1' for unique writer).
    """

    def __init__(self, artifact_path):
        """
        Initializes a new instance of the Source class.

        Parameters
        ----------
        artifact_path : str
            The path to the data.
        """

        self.artifact_path = artifact_path
        self.base_path = os.path.join(self.artifact_path, 'iam')

        self.partition_path = os.path.join(self.base_path, 'largeWriterIndependentTextLineRecognitionTask')
        self.training_file_path = os.path.join(self.partition_path, 'trainset.txt')
        self.validation_file_path = os.path.join(self.partition_path, 'validationset1.txt')
        self.test_file_path = os.path.join(self.partition_path, 'testset.txt')

        self.transcription_path = os.path.join(self.base_path, 'ascii')
        self.words_file_path = os.path.join(self.transcription_path, 'words.txt')
        self.lines_file_path = os.path.join(self.transcription_path, 'lines.txt')
        self.forms_file_path = os.path.join(self.transcription_path, 'forms.txt')

    def fetch_data(self, text_level):
        """
        Retrieves the data for training, validation, and test partitions.

        Parameters
        ----------
        text_level : str
            The granularity of the data to be retrieved.

        Returns
        -------
        dict
            Partition dictionary with list of items.
        """

        data = {'training': [], 'validation': [], 'test': []}

        if not os.path.isdir(self.base_path):
            return data

        training_partition_data = self._load_partition_data(self.training_file_path)
        validation_partition_data = self._load_partition_data(self.validation_file_path)
        test_partition_data = self._load_partition_data(self.test_file_path)

        if text_level == 'word':
            words_data = self._load_words_data(self.words_file_path)

            data['training'] = self._filter_data(words_data, training_partition_data)
            data['validation'] = self._filter_data(words_data, validation_partition_data)
            data['test'] = self._filter_data(words_data, test_partition_data)

        elif text_level == 'line':
            lines_data = self._load_lines_data(self.lines_file_path)

            data['training'] = self._filter_data(lines_data, training_partition_data)
            data['validation'] = self._filter_data(lines_data, validation_partition_data)
            data['test'] = self._filter_data(lines_data, test_partition_data)

        elif text_level == 'paragraph':
            lines_data = self._load_lines_data(self.lines_file_path, bbox_info=True)
            paragraphs_data = self._load_paragraphs_data(lines_data, bbox_info=True)

            data['training'] = self._filter_data(paragraphs_data, training_partition_data, form_group=True)
            data['validation'] = self._filter_data(paragraphs_data, validation_partition_data, form_group=True)
            data['test'] = self._filter_data(paragraphs_data, test_partition_data, form_group=True)

        return data

    def _load_partition_data(self, file_path):
        """
        Loads the partition data.

        Parameters
        ----------
        file_path : str
            The path to the file containing the partition data.

        Returns
        -------
        list
            A list of partition data.
        """

        with open(file_path, 'r') as file:
            data = [row.strip() for row in file.readlines()]

        return data

    def _filter_data(self, data, partition_data, form_group=False):
        """
        Filter the given data based on the partition data using multiprocessing.

        Parameters
        ----------
        data : list
            The data to filter.
        partition_data : list
            The partition data to match against.
        form_group : bool, optional
            Indicates whether to group the image ID.

        Returns
        -------
        list
            The filtered data that matches the partition data.
        """

        with multiprocessing.get_context('fork').Pool() as pool:
            arguments = [(x, partition_data, form_group) for x in data]
            data = [x for x in pool.starmap(self._validate_filtered_data, arguments) if x]

        return data

    def _validate_filtered_data(self, item, partition_data, form_group=False):
        """
        Validate if the given item matches any partition data.

        Parameters
        ----------
        item : any
            The item to validate.
        partition_data : list
            The partition data to match against.
        form_group : bool, optional
            Indicates whether to group the image ID.

        Returns
        -------
        item or None
            The item if it matches any partition data, or None if there is no match.
        """

        filtered_data = None

        for image_id in partition_data:
            if form_group:
                image_id = '-'.join(image_id.split('-')[:2])

            if image_id in item['image']:
                filtered_data = item

        return filtered_data

    def _get_writers(self):
        """
        Parses a file to assign unique IDs to each writer.

        Returns
        -------
        dict
            Maps form IDs to unique writer IDs.
        """

        writers = {}

        with open(self.forms_file_path, 'r') as forms_file:
            for line in forms_file:
                if line.startswith('#'):
                    continue

                parts = line.strip().split()
                if len(parts) >= 2:
                    form_id, writer_id = parts[0], parts[1]
                    writers[form_id] = writer_id

        return writers

    def _load_words_data(self, file_path, bbox_info=False):
        """
        Loads the words data.

        Parameters
        ----------
        file_path : str
            The path to the file containing the words data.
        bbox_info : bool, optional
            Determines whether to include bounding box information in the words data.

        Returns
        -------
        list
            A list of words data.
        """

        words_data = []
        writers = self._get_writers()

        with open(file_path, 'r') as file:
            rows = file.readlines()

        for row in rows:
            if row.startswith('#'):
                continue

            parts = row.split()
            word_ids = parts[0].split('-')
            word_path = os.path.join('words', word_ids[0], '-'.join(word_ids[:2]))
            word_file_name = f"{parts[0]}.png"

            image_path = os.path.join(self.base_path, word_path, word_file_name)
            bbox = [int(x) for x in parts[4:8]] if bbox_info else []
            text = parts[-1]

            form_id = '-'.join(word_ids[:2])
            writer_id = writers.get(form_id, '1')

            words_data.append({
                'image': image_path,
                'bbox': bbox,
                'text': text,
                'writer': writer_id,
            })

        return words_data

    def _load_lines_data(self, file_path, bbox_info=False):
        """
        Loads the lines data.

        Parameters
        ----------
        file_path : str
            The path to the file containing the lines data.
        bbox_info : bool, optional
            Determines whether to include bounding box information in the lines data.

        Returns
        -------
        list
            A list of lines data.
        """

        lines_data = []
        writers = self._get_writers()

        with open(file_path, 'r') as file:
            rows = file.readlines()

        for row in rows:
            if row.startswith('#'):
                continue

            parts = row.split()
            line_ids = parts[0].split('-')
            line_path = os.path.join('lines', line_ids[0], '-'.join(line_ids[:2]))
            line_file_name = f"{parts[0]}.png"

            image_path = os.path.join(self.base_path, line_path, line_file_name)
            bbox = [int(x) for x in parts[4:8]] if bbox_info else []
            text = ' '.join(parts[8:]).replace('|', ' ')

            form_id = '-'.join(line_ids[:2])
            writer_id = writers.get(form_id, '1')

            lines_data.append({
                'image': image_path,
                'bbox': bbox,
                'text': text,
                'writer': writer_id,
            })

        return lines_data

    def _load_paragraphs_data(self, lines_data, bbox_info=False):
        """
        Loads the paragraphs data.

        Parameters
        ----------
        lines_data : str
            The lines data as base content to paragraph data construction.
        bbox_info : bool, optional
            Determines whether to include bounding box information in the paragraphs data.

        Returns
        -------
        list
            A list of paragraphs data.
        """

        paragraphs_data = {}

        for line in lines_data:
            parts = os.path.basename(line['image']).split('-')
            pattern = os.path.join(self.base_path, "forms**", f"{'-'.join(parts[:2])}.png")
            file_path = next(iter(glob.glob(pattern, recursive=True)), None)

            if not file_path:
                continue

            bbox = line['bbox']
            new_bbox = [
                bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]
            ]

            paragraph = paragraphs_data.setdefault(file_path, {
                "image": file_path,
                "bbox": new_bbox,
                "text": "",
                "writer": line['writer'],
            })

            paragraph['bbox'] = [
                min(paragraph['bbox'][0], new_bbox[0]),
                min(paragraph['bbox'][1], new_bbox[1]),
                max(paragraph['bbox'][2], new_bbox[2]),
                max(paragraph['bbox'][3], new_bbox[3])
            ]

            paragraph['text'] += (f"\n{line['text']}" if paragraph['text'] else line['text'])

        paragraphs_data = list(paragraphs_data.values())

        if bbox_info:
            for paragraph in paragraphs_data:
                bbox = paragraph['bbox']
                paragraph['bbox'] = [
                    bbox[0],
                    bbox[1],
                    abs(bbox[2] - bbox[0]),
                    abs(bbox[3] - bbox[1])
                ]
        else:
            for paragraph in paragraphs_data:
                paragraph['bbox'] = []

        return paragraphs_data
