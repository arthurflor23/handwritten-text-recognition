import os
import glob
import multiprocessing


class Source():
    """
    Represents the BRESSAY database source.

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
        self.base_path = os.path.join(self.artifact_path, 'bressay')

        self.data_path = os.path.join(self.base_path, 'data')
        self.sets_path = os.path.join(self.base_path, 'sets')

        self.training_file_path = os.path.join(self.sets_path, 'training.txt')
        self.validation_file_path = os.path.join(self.sets_path, 'validation.txt')
        self.test_file_path = os.path.join(self.sets_path, 'test.txt')

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

        training_partition_data = self._load_partition_data(self.training_file_path)
        validation_partition_data = self._load_partition_data(self.validation_file_path)
        test_partition_data = self._load_partition_data(self.test_file_path)

        data_path = os.path.join(self.data_path, f'{text_level}s')

        data['training'] = self._load_data(data_path, training_partition_data)
        data['validation'] = self._load_data(data_path, validation_partition_data)
        data['test'] = self._load_data(data_path, test_partition_data)

        return data

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

    def _load_data(self, data_path, partition_data):
        """
        Load data for a given partition.

        Parameters
        ----------
        data_path : str
            The path to the data directory.
        partition_data : list
            List of page IDs in the partition.

        Returns
        -------
        list
            List of data entries.
        """

        with multiprocessing.get_context('fork').Pool() as pool:
            arguments = [(data_path, x) for x in partition_data]
            data = [y for x in pool.starmap(self._load_files, arguments) for y in x]

        return data

    def _load_files(self, data_path, page_id):
        """
        Loads content files from a page.

        Parameters
        ----------
        data_path : str
            The path to the data directory.
        page_id : str
            The ID of the page to load the content.

        Returns
        -------
        list
            List of data entries from the page.
        """

        data_path = os.path.join(data_path, page_id)
        image_files = glob.glob(os.path.join(data_path, '**', '*.png'), recursive=True)

        data = []
        for image_file in image_files:
            text_file = image_file.replace('.png', '.txt')

            with open(text_file, 'r') as file:
                text = file.read().strip()

            data.append({
                'image': image_file,
                'bbox': [],
                'text': text,
                'writer': page_id,
            })

        return data
