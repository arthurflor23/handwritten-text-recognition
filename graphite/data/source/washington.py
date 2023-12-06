import os


class Source():
    """
    Represents the Washington database source.

    Requires implementation of `fetch_data`, returning a dictionary with
        'training', 'validation', and 'test' keys, each mapping to a list of data entries.

    Each data entry is a dictionary with keys 'image', 'bbox', 'text', and 'writer':
        'image' : str
            path to the image.
        'bbox' : list
            bounding box coordinates [x, y, h, w] (empty if no bbox).
        'text' : str
            text content, with '\n' as line break.
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
        self.base_path = os.path.join(self.artifact_path, 'washington')

        self.partition_path = os.path.join(self.base_path, 'sets', 'cv1')
        self.training_file_path = os.path.join(self.partition_path, 'train.txt')
        self.validation_file_path = os.path.join(self.partition_path, 'valid.txt')
        self.test_file_path = os.path.join(self.partition_path, 'test.txt')

        self.transcription_path = os.path.join(self.base_path, 'ground_truth')
        self.words_file_path = os.path.join(self.transcription_path, 'word_labels.txt')
        self.lines_file_path = os.path.join(self.transcription_path, 'transcription.txt')

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

    def _filter_data(self, data, partition_data):
        """
        Filter the given data based on the partition data.

        Parameters
        ----------
        data : list
            The data to filter.
        partition_data : list
            The partition data to match against.

        Returns
        -------
        list
            The filtered data that matches the partition data.
        """

        filtered_data = []

        for image_id in partition_data:
            for item in data:
                if image_id in item['image']:
                    filtered_data.append(item)

        return filtered_data

    def _load_words_data(self, file_path):
        """
        Loads the words data from a file.

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

        with open(file_path, 'r') as file:
            rows = file.readlines()

        for row in rows:
            if row.startswith('#'):
                continue

            parts = row.split()
            word_path = os.path.join(self.base_path, 'data', 'word_images_normalized')
            word_file_name = f"{parts[0]}.png"

            image_path = os.path.join(word_path, word_file_name)
            text = self._format_label(parts[1]).replace('-', '').replace('|', ' ')

            words_data.append({
                'image': image_path,
                'bbox': [],
                'text': text,
                'writer': '1',
            })

        return words_data

    def _load_lines_data(self, file_path):
        """
        Loads the lines data from a file.

        Parameters
        ----------
        file_path : str
            The path to the file containing the lines data.

        Returns
        -------
        list
            A list of lines data.
        """

        lines_data = []

        with open(file_path, 'r') as file:
            rows = file.readlines()

        for row in rows:
            if row.startswith('#'):
                continue

            parts = row.split()
            line_path = os.path.join(self.base_path, 'data', 'line_images_normalized')
            line_file_name = f"{parts[0]}.png"

            image_path = os.path.join(line_path, line_file_name)
            text = self._format_label(parts[1]).replace('-', '').replace('|', ' ')

            lines_data.append({
                'image': image_path,
                'bbox': [],
                'text': text,
                'writer': '1',
            })

        return lines_data

    def _format_label(self, label):
        """
        Standardizes a label.

        Parameters
        ----------
        label : str
            The label to be standardized.

        Returns
        -------
        str
            The standardized label.
        """

        substitutions = {
            's_pt': '.',
            's_cm': ',',
            's_mi': '-',
            's_qo': ':',
            's_sq': ';',
            's_et': 'v',
            's_bl': '(',
            's_br': ')',
            's_qt': '\'',
            's_GW': 'G.W.',
            's_s': 's',         # ſ
            's_': '',
        }

        for pattern, replacement in substitutions.items():
            label = label.replace(pattern, replacement)

        return label
