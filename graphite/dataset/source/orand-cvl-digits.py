import os
import glob


class Source():
    """
    Represents the CVL Digits database source.
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
        self.base_path = os.path.join(self.data_path, 'orand-car-2014', 'cvl-digits')

        self.training_path = os.path.join(self.base_path, 'cvl-strings', 'train', '**', '*.png')
        self.test_path = os.path.join(self.base_path, 'cvl-strings-eval', '**', '*.png')

    def get_line_data(self):
        """
        Retrieves the line data for training, validation, and testing.

        Returns
        -------
        tuple
            A tuple containing lists of training, validation, and test lines data.
        """

        def process_file(file_path):
            base_name = os.path.basename(file_path)
            label = base_name.split('-')[0]
            return [file_path, [], label]

        training_files = glob.glob(self.training_path, recursive=True)
        training_lines = [process_file(file_path) for file_path in training_files]

        test_files = glob.glob(self.test_path, recursive=True)
        test_lines = [process_file(file_path) for file_path in test_files]

        return training_lines, [], test_lines
