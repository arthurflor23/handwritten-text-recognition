import os
import sys
import importlib


class Source():
    """
    Represents all data sources.

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

        curr_file_name = os.path.basename(__file__)
        curr_dir = os.path.dirname(os.path.realpath(__file__))

        sys.path.append(curr_dir)

        filenames = [
            filename for filename in os.listdir(curr_dir)
            if filename.endswith('.py') and filename not in {curr_file_name, '__init__.py'}
        ]

        data = {'training': [], 'validation': [], 'test': []}

        for filename in filenames:
            module = importlib.import_module(filename.rstrip('.py'))
            source = module.Source(self.artifact_path)

            source_data = source.fetch_data(text_level)

            data['training'].extend(source_data['training'])
            data['validation'].extend(source_data['validation'])
            data['test'].extend(source_data['test'])

        return data
