import os


class Source():
    """
    Represents the IAM database source.
    """

    def __init__(self, data_path):
        """
        Initializes a new instance of the Source class.

        Parameters
        ----------
        data_path : str
            The path to the data.

        Returns
        -------
        None
        """

        self.data_path = data_path
        self.base_path = os.path.join(self.data_path, 'iam')

        self.transcription_path = os.path.join(self.base_path, 'ascii')
        self.partition_path = os.path.join(self.base_path, 'largeWriterIndependentTextLineRecognitionTask')

        self.training_file_path = os.path.join(self.partition_path, 'trainset.txt')
        self.validation_file_path = os.path.join(self.partition_path, 'validationset1.txt')
        self.test_file_path = os.path.join(self.partition_path, 'testset.txt')

        self.words_file_path = os.path.join(self.transcription_path, 'words.txt')
        self.lines_file_path = os.path.join(self.transcription_path, 'lines.txt')

    def get_line_data(self):
        """
        Retrieves the line data for training, validation, and testing.

        Returns
        -------
        tuple
            A tuple containing lists of training, validation, and test lines data.
        """

        training_partition_data = self._load_partition_data(self.training_file_path)
        validation_partition_data = self._load_partition_data(self.validation_file_path)
        test_partition_data = self._load_partition_data(self.test_file_path)

        lines_data = self._load_lines_data(self.lines_file_path)

        training_lines = self._filter_lines_data(lines_data, training_partition_data)
        validation_lines = self._filter_lines_data(lines_data, validation_partition_data)
        test_lines = self._filter_lines_data(lines_data, test_partition_data)

        return training_lines, validation_lines, test_lines

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
            data = [line.strip() for line in file.readlines()]

        return data

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
            lines = file.readlines()

        lines_data = []

        for line in lines:
            if line.startswith('#'):
                continue

            parts = line.split()

            line_ids = parts[0].split('-')
            line_path = os.path.join('lines', line_ids[0], '-'.join(line_ids[:2]))
            line_file_name = f"{parts[0]}.png"

            image_path = os.path.join(self.base_path, line_path, line_file_name)
            bbox = []  # [int(x) for x in parts[6:8] + parts[4:6]]
            label = ' '.join(parts[8:]).replace('|', ' ')

            lines_data.append([image_path, bbox, label])

        return lines_data

    def _filter_lines_data(self, lines_data, partition_data):
        """
        Filters the lines data based on the partition data.

        Parameters
        ----------
        lines_data : list
            The lines data to filter.

        partition_data : list
            The partition data to use for filtering.

        Returns
        -------
        list
            The filtered lines data.
        """

        filtered_data = []

        for line_id in partition_data:
            for line_data in lines_data:
                if line_id in line_data[0]:
                    filtered_data.append(line_data)
                    break

        return filtered_data
