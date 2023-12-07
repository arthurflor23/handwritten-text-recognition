import os


class Source():
    """
    Represents the ORAND-CAR-B-2014 database source.

    Requires implementation of `fetch_data`, returning a dictionary with
        'training', 'validation', and 'test' keys, each mapping to a list of data entries.

    Each data entry is a dictionary with keys 'image', 'bbox', 'text', and 'writer':
        'image' : str
            path to the image.
        'bbox' : list
            bounding box coordinates [x, y, h, w] (empty if no bbox).
        'text' : str
            text content, with '\n' and '\n\n' as line and paragraph breaks.
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
        self.base_path = os.path.join(self.artifact_path, 'orand-car')

        self.training_path = os.path.join(self.base_path, 'CAR-B', 'b_train_images')
        self.test_path = os.path.join(self.base_path, 'CAR-B', 'b_test_images')

        self.training_file_path = os.path.join(self.base_path, 'CAR-B', 'b_train_gt.txt')
        self.test_file_path = os.path.join(self.base_path, 'CAR-B', 'b_test_gt.txt')

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

        def process_row(row, file_path):
            row = row.strip().split('\t')
            path = os.path.join(file_path, row[0])

            return {
                'image': path,
                'bbox': [],
                'text': ' '.join(list(row[1])),
                'writer': '1',
            }

        if text_level == 'word' or text_level == 'line':
            with open(self.training_file_path, 'r') as training_file:
                data['training'] = [process_row(row, self.training_path) for row in training_file.readlines()]

            with open(self.test_file_path, 'r') as test_file:
                data['test'] = [process_row(row, self.test_path) for row in test_file.readlines()]

        return data
