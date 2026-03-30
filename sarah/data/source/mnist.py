import os
import glob


class Source():
    """
    Represents the MNIST database source.

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
        self.base_path = os.path.join(self.artifact_path, 'mnist')

        self.training_path = os.path.join(self.base_path, 'training')
        self.test_path = os.path.join(self.base_path, 'test')

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

        data['training'] = self._load_data(self.training_path)
        data['test'] = self._load_data(self.test_path)

        return data

    def _load_data(self, partition_path):
        """
        Loads the partition data.

        Parameters
        ----------
        partition_path : str
            The partition path.

        Returns
        -------
        list
            A list of partition items.
        """

        if not os.path.isdir(partition_path):
            return []

        image_paths = sorted(glob.glob(os.path.join(partition_path, '*.png')))
        partition_data = []

        for image_path in image_paths:
            image_file_name = os.path.basename(image_path)
            text = os.path.splitext(image_file_name)[0].split('-')[-1]

            partition_data.append({
                'image': image_path,
                'bbox': [],
                'text': text,
                'writer': '1',
            })

        return partition_data
