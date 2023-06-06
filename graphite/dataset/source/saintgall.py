import os


class Source():
    """
    Represents the Saint Gall database source.
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
        self.base_path = os.path.join(self.artifact_path, 'saintgall')

        self.partition_path = os.path.join(self.base_path, 'sets')
        self.training_file_path = os.path.join(self.partition_path, 'train.txt')
        self.validation_file_path = os.path.join(self.partition_path, 'valid.txt')
        self.test_file_path = os.path.join(self.partition_path, 'test.txt')

        self.transcription_path = os.path.join(self.base_path, 'ground_truth')
        self.lines_file_path = os.path.join(self.transcription_path, 'transcription.txt')

    def fetch_data(self, _):
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
            label = parts[1].replace('-', '').replace('|', ' ')

            # Construct the line data entry with image path, bounding box, and label
            lines_data.append([image_path, bbox, label])

        return lines_data
