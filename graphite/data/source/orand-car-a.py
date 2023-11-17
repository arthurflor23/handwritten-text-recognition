import os


class Source():
    """
    Represents the ORAND-CAR-A-2014 database source.
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

        self.training_path = os.path.join(self.base_path, 'CAR-A', 'a_train_images')
        self.test_path = os.path.join(self.base_path, 'CAR-A', 'a_test_images')

        self.training_file_path = os.path.join(self.base_path, 'CAR-A', 'a_train_gt.txt')
        self.test_file_path = os.path.join(self.base_path, 'CAR-A', 'a_test_gt.txt')

    def fetch_data(self, text_level):
        """
        Retrieves the data for training, validation, and testing.

        Parameters
        ----------
        text_level : str
            The granularity of the data to be retrieved.

        Returns
        -------
        tuple
            A tuple containing lists of training, validation, and test data.
        """

        def process_row(row, file_path):
            row = row.strip().split('\t')

            path = os.path.join(file_path, row[0])
            label = ' '.join(list(row[1]))

            return [path, [], label]

        training_data, validation_data, test_data = [], [], []

        if text_level == 'word' or text_level == 'line':
            with open(self.training_file_path, 'r') as training_file:
                training_data = [process_row(row, self.training_path) for row in training_file.readlines()]

            with open(self.test_file_path, 'r') as test_file:
                test_data = [process_row(row, self.test_path) for row in test_file.readlines()]

        return training_data, validation_data, test_data
