import os
import sys
import importlib


class Source():
    """
    Represents all data sources.
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

        curr_file_name = os.path.basename(__file__)
        curr_dir = os.path.dirname(os.path.realpath(__file__))

        sys.path.append(curr_dir)

        filenames = [
            filename for filename in os.listdir(curr_dir)
            if filename.endswith('.py') and filename not in {curr_file_name, '__init__.py'}
        ]

        training_data, validation_data, test_data = [], [], []

        for filename in filenames:
            module = importlib.import_module(filename.rstrip('.py'))
            source = module.Source(self.artifact_path)

            train, valid, test = source.fetch_data(text_level)

            training_data.extend(train)
            validation_data.extend(valid)
            test_data.extend(test)

        return training_data, validation_data, test_data
