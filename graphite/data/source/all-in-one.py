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

        curr_dir = os.path.dirname(os.path.realpath(__file__))
        sys.path.append(curr_dir)

        include_list = [
            'bentham',
            'iam',
            'parzival',
            'rimes',
            'saintgall',
            'washington',
        ]

        filenames = [x for x in os.listdir(curr_dir) if x.rstrip('.py') in include_list]
        filenames.sort()

        data = {'training': [], 'validation': [], 'test': []}

        global_writer_mapping = {}
        global_writer_id = 1

        for filename in filenames:
            module = importlib.import_module(filename[:-3])
            source = module.Source(self.artifact_path)

            source_data = source.fetch_data(text_level)

            local_writer_mapping = {}
            local_writer_id = 1

            for dataset_type in ['training', 'validation', 'test']:
                for item in source_data[dataset_type]:
                    original_writer = item['writer']

                    if original_writer not in local_writer_mapping:
                        local_writer_mapping[original_writer] = local_writer_id
                        global_writer_mapping[local_writer_id] = global_writer_id

                        local_writer_id += 1
                        global_writer_id += 1

                    item['writer'] = str(global_writer_mapping[local_writer_mapping[original_writer]])

                data[dataset_type].extend(source_data[dataset_type])

        return data
