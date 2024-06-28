import os
import glob


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

        self.writers = self._get_writers(self.forms_file_path)

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

        training_partition_data = self._read_file(self.training_file_path)
        validation_partition_data = self._read_file(self.validation_file_path)
        test_partition_data = self._read_file(self.test_file_path)

        if text_level == 'word':
            words_data = self._load_words_data(self.words_file_path)

            data['training'] = self._get_data(words_data, training_partition_data)
            data['validation'] = self._get_data(words_data, validation_partition_data)
            data['test'] = self._get_data(words_data, test_partition_data)

        elif text_level == 'line':
            lines_data = self._load_lines_data(self.lines_file_path)

            data['training'] = self._get_data(lines_data, training_partition_data)
            data['validation'] = self._get_data(lines_data, validation_partition_data)
            data['test'] = self._get_data(lines_data, test_partition_data)

        elif text_level == 'paragraph':
            lines_data = self._load_lines_data(self.lines_file_path)
            paragraphs_data = self._load_paragraphs_data(lines_data)

            data['training'] = self._get_data(paragraphs_data, training_partition_data, group=True)
            data['validation'] = self._get_data(paragraphs_data, validation_partition_data, group=True)
            data['test'] = self._get_data(paragraphs_data, test_partition_data, group=True)

        return data

    def _read_file(self, file_path):
        """
        Loads the data content from file.

        Parameters
        ----------
        file_path : str
            The path to the file content.

        Returns
        -------
        list
            A list of rows.
        """

        with open(file_path, 'r') as f:
            data = [x.strip() for x in f.readlines() if not x.startswith('#')]

        return data

    def _get_data(self, data, partition_data, group=False):
        """
        Filter the given data based on partition data.

        Parameters
        ----------
        data : list of dict
            The data to filter, each with an 'id' key.
        partition_data : list
            The IDs to match against.
        group : bool, optional
            Indicates whether to group the image ID.

        Returns
        -------
        list of dict
            The filtered data matching the partition IDs.
        """

        if group:
            partition_data = ['-'.join(x.split('-')[:2]) for x in partition_data]

        filtered = [x for x in data if x['id'] in partition_data]

        return filtered

    def _get_writers(self, forms_file_path):
        """
        Parses a file to assign unique IDs to each writer.

        Parameters
        ----------
        forms_file_path : str
            The forms filepath with the writers IDs.

        Returns
        -------
        dict
            Maps form IDs to unique writer IDs.
        """

        writers = {}
        data = self._read_file(forms_file_path)

        for row in data:
            row = row.split()

            if len(row) > 1:
                form_id, writer_id = row[0], row[1]
                writers[form_id] = writer_id

        return writers

    def _load_words_data(self, file_path):
        """
        Loads the words data.

        Parameters
        ----------
        file_path : str
            The path to the file containing the words data.

        Returns
        -------
        list
            A list of words data.
        """

        words_data = []
        data = self._read_file(file_path)

        for row in data:
            row = row.split()

            ids = row[0].split('-')
            form_id = '-'.join(ids[:2])
            item_id = '-'.join(ids[:3])
            writer_id = self.writers.get(form_id, '1')
            bbox = []
            text = ' '.join(row[-1].replace('|', ' ').split())

            word_path = os.path.join('words', ids[0], form_id, f"{row[0]}.png")
            image_path = os.path.join(self.base_path, word_path)

            words_data.append({
                'id': item_id,
                'image': image_path,
                'bbox': bbox,
                'text': text,
                'writer': writer_id,
            })

        return words_data

    def _load_lines_data(self, file_path):
        """
        Loads the lines data.

        Parameters
        ----------
        file_path : str
            The path to the file containing the lines data.

        Returns
        -------
        list
            A list of lines data.
        """

        exclude_data = ['p02-109-01']

        lines_data = []
        data = self._read_file(file_path)

        for row in data:
            row = row.split()

            ids = row[0].split('-')
            form_id = '-'.join(ids[:2])
            item_id = '-'.join(ids[:3])

            if item_id in exclude_data:
                continue

            writer_id = self.writers.get(form_id, '1')
            bbox = [int(x) for x in row[4:8]]
            text = ' '.join(row[-1].replace('|', ' ').split())

            pattern = os.path.join(self.base_path, "forms**", f"{form_id}.png")
            image_path = next(iter(glob.glob(pattern, recursive=True)), None)

            lines_data.append({
                'id': item_id,
                'image': image_path,
                'bbox': bbox,
                'text': text,
                'writer': writer_id,
            })

        return lines_data

    def _load_paragraphs_data(self, lines_data):
        """
        Loads the paragraphs data.

        Parameters
        ----------
        lines_data : list
            Line data as base content for paragraph data.

        Returns
        -------
        list
            A list of paragraphs data.
        """

        paragraphs_data = {}

        for line in lines_data:
            ids = os.path.basename(line['id']).split('-')
            form_id = '-'.join(ids[:2])

            pattern = os.path.join(self.base_path, "forms**", f"{form_id}.png")
            image_path = next(iter(glob.glob(pattern, recursive=True)), None)

            bbox = [
                line['bbox'][0],
                line['bbox'][1],
                line['bbox'][2] + line['bbox'][0],
                line['bbox'][3] + line['bbox'][1],
            ]

            content = {
                'id': form_id,
                'image': image_path,
                'bbox': bbox,
                'text': '',
                'writer': line['writer'],
            }

            paragraph = paragraphs_data.setdefault(image_path, content)

            paragraph['bbox'] = [
                min(paragraph['bbox'][0], bbox[0]),
                min(paragraph['bbox'][1], bbox[1]),
                max(paragraph['bbox'][2], bbox[2]),
                max(paragraph['bbox'][3], bbox[3])
            ]

            paragraph['text'] += (f"\n{line['text']}" if paragraph['text'] else line['text'])

        paragraphs_data = list(paragraphs_data.values())

        for paragraph in paragraphs_data:
            paragraph['bbox'] = [
                paragraph['bbox'][0],
                paragraph['bbox'][1],
                abs(paragraph['bbox'][2] - paragraph['bbox'][0]),
                abs(paragraph['bbox'][3] - paragraph['bbox'][1])
            ]

        return paragraphs_data
