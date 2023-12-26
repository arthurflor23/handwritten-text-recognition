import os
import glob


class Source():
    """
    Represents the CVL Digits database source.

    Requires implementation of `fetch_data`, returning a dictionary with
        'training', 'validation', and 'test' keys, each mapping to a list of data entries.

    Each data entry is a dictionary with keys 'image', 'bbox', 'text', and 'writer':
        'image' : str
            path to the image.
        'bbox' : list
            bounding box coordinates [x, y, h, w] (empty if no bbox).
        'text' : str
            text content with '\n' as line breaks.
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
        self.base_path = os.path.join(self.artifact_path, 'cvl-digits')

        self.training_path = os.path.join(self.base_path, 'cvl-strings', 'train', '**', '*.png')
        self.test_path = os.path.join(self.base_path, 'cvl-strings-eval', '**', '*.png')

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

        if not os.path.isdir(self.base_path):
            return data

        def process_file(file_path):
            base_name = os.path.basename(file_path)
            name_part = base_name.split('-')

            return {
                'image': file_path,
                'bbox': [],
                'text': ' '.join(list(name_part[0])),
                'writer': name_part[1],
            }

        if text_level == 'word' or text_level == 'line':
            training_files = glob.glob(self.training_path, recursive=True)
            data['training'] = [process_file(file_path) for file_path in training_files]

            test_files = glob.glob(self.test_path, recursive=True)
            data['test'] = [process_file(file_path) for file_path in test_files]

        return data
