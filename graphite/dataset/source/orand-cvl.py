import os
import glob


class Source():
    """
    Represents the CVL Digits database source.
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
        self.base_path = os.path.join(self.artifact_path, 'orand-cvl')

        self.training_path = os.path.join(self.base_path, 'cvl-strings', 'train', '**', '*.png')
        self.test_path = os.path.join(self.base_path, 'cvl-strings-eval', '**', '*.png')

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

        def process_file(file_path):
            base_name = os.path.basename(file_path)
            label = base_name.split('-')[0]
            return [file_path, [], label]

        training_files = glob.glob(self.training_path, recursive=True)
        training_data = [process_file(file_path) for file_path in training_files]

        test_files = glob.glob(self.test_path, recursive=True)
        test_data = [process_file(file_path) for file_path in test_files]

        return training_data, [], test_data
