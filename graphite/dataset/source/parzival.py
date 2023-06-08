import os


class Source():
    """
    Represents the Parzival database source.
    """

    def __init__(self, artifact_path):
        """
        Initializes a new instance of the Source class.

        Parameters
        ----------
        artifact_path : str
            The path to the data.

        Returns
        -------
        None
        """

        self.artifact_path = artifact_path
        self.base_path = os.path.join(self.artifact_path, 'parzival')

        self.partition_path = os.path.join(self.base_path, 'sets1')
        self.training_file_path = os.path.join(self.partition_path, 'train.txt')
        self.validation_file_path = os.path.join(self.partition_path, 'valid.txt')
        self.test_file_path = os.path.join(self.partition_path, 'test.txt')

        self.transcription_path = os.path.join(self.base_path, 'ground_truth')
        self.words_file_path = os.path.join(self.transcription_path, 'word_labels.txt')
        self.lines_file_path = os.path.join(self.transcription_path, 'transcription.txt')

    def fetch_data(self, level):
        """
        Retrieves the data for training, validation, and testing.

        Parameters
        ----------
        level : str
            The granularity level of the data to be retrieved.

        Returns
        -------
        tuple
            A tuple containing lists of training, validation, and test data.
        """

        # Load the partition data for training, validation, and testing
        training_partition_data = self._load_partition_data(self.training_file_path)
        validation_partition_data = self._load_partition_data(self.validation_file_path)
        test_partition_data = self._load_partition_data(self.test_file_path)

        if level == 'word':
            # Load the words data from the file
            words_data = self._load_words_data(self.words_file_path)

            # Filter the words data based on the partition data
            training_data = self._filter_data(words_data, training_partition_data)
            validation_data = self._filter_data(words_data, validation_partition_data)
            test_data = self._filter_data(words_data, test_partition_data)

        elif level == 'line':
            # Load the lines data from the file
            lines_data = self._load_lines_data(self.lines_file_path)

            # Filter the lines data based on the partition data
            training_data = self._filter_data(lines_data, training_partition_data)
            validation_data = self._filter_data(lines_data, validation_partition_data)
            test_data = self._filter_data(lines_data, test_partition_data)

        return training_data, validation_data, test_data

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
                if image_id in item[0]:
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

        with open(file_path, 'r') as file:
            rows = file.readlines()

        words_data = []

        for row in rows:
            if row.startswith('#'):
                continue

            parts = row.split()
            word_path = os.path.join(self.base_path, 'data', 'word_images_normalized')
            word_file_name = f"{parts[0]}.png"

            image_path = os.path.join(word_path, word_file_name)
            bbox = []
            label = self._format_label(parts[1]).replace('-', '').replace('|', ' ')

            words_data.append([image_path, bbox, label])

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

        with open(file_path, 'r') as file:
            rows = file.readlines()

        lines_data = []

        for row in rows:
            if row.startswith('#'):
                continue

            parts = row.split()
            line_path = os.path.join(self.base_path, 'data', 'line_images_normalized')
            line_file_name = f"{parts[0]}.png"

            image_path = os.path.join(line_path, line_file_name)
            bbox = []
            label = self._format_label(parts[1]).replace('-', '').replace('|', ' ')

            lines_data.append([image_path, bbox, label])

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
            'pt': '.',
            'eq': '-',
            'ha115': 's',
            'hc097': 'æ',
            'hp115': 'ſ',
            'hs101079': 'O',    # ǒ
            'hs101105': 'i',
            'hs101111': 'œ',
            'hs101118': 'v',    # v̊
            'hs111117': 'u',    # ů
            'hs111118': 'v',    # v̊
            'hs111119': 'w',    # ẘ
            'hs118101': 'e',    # ě
            'hs118105': 'i',    # ǐ
            'hs118111': 'o',    # ǒ
            'p046117': 'u',
            'p046121': 'y',
            'p047097': 'a',     # á
            'p047101': 'e',     # é
            'p047111': 'o',     # ó
            'p047117': 'u',
            'p047118': 'v',
            'p060079': 'O',
            'p060097': 'a',     # â
            'p060101': 'e',     # ê
            'p060105': 'i',     # î
            'p060111': 'o',     # ô
            'p060118': 'u',     # û
            'sa1m1': 'a',       # ā
            'sa1n1': 'a',       # ã
            'sd03e03': 'E',     # ɛ
            'sd7er7': '§',
            'se1n1': 'e',       # ẽ
            'sg9ra9': 'g',      # ḡ
            'sh7er7': 'h',
            'si1n1': 'i',       # ĩ
            'si4us4': 'j',      # į
            'sm4us4': 'm',      # m̂
            'sm7er7': 'm',      # ḿ
            'sn1d1': 'n',       # n̄
            'so1n1': 'o',       # õ
            'sp09ri09': 'p',    # p̄
            'sp5er5': 'p',
            'sp9ra9': 'p',      # p̃
            'st04us04': 't',    # ẗ
            'st7er7': 't',      # t̄
            'su1m1': 'u',       # ū
            'sv1n1': 'v',       # v̄
            'sv7er7': 'v',      # ṽ
        }

        for pattern, replacement in substitutions.items():
            label = label.replace(pattern, replacement)

        return label
