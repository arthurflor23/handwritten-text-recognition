import os
import glob
import multiprocessing
import xml.etree.ElementTree as ET


class Source():
    """
    Represents the Bentham database source.

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
        self.base_path = os.path.join(self.artifact_path, 'bentham')

        self.partition_path = os.path.join(self.base_path, 'Partitions')
        self.training_file_path = os.path.join(self.partition_path, 'Train.lst')
        self.validation_file_path = os.path.join(self.partition_path, 'Validation.lst')
        self.test_file_path = os.path.join(self.partition_path, 'Test.lst')

        self.lines_file_path = os.path.join(self.base_path, 'Transcriptions', '**.txt')
        self.paragraphs_file_path = os.path.join(self.base_path, 'PAGE', '**.xml')

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

        if text_level == 'line':
            lines_data = self._load_lines_data(self.lines_file_path)

            data['training'] = self._filter_data(lines_data, training_partition_data)
            data['validation'] = self._filter_data(lines_data, validation_partition_data)
            data['test'] = self._filter_data(lines_data, test_partition_data)

        elif text_level == 'paragraph':
            paragraphs_data = self._load_paragraphs_data(self.paragraphs_file_path)

            data['training'] = self._filter_data(paragraphs_data, training_partition_data)
            data['validation'] = self._filter_data(paragraphs_data, validation_partition_data)
            data['test'] = self._filter_data(paragraphs_data, test_partition_data)

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
            data = [row.strip() for row in file.readlines()]

        return data

    def _filter_data(self, data, partition_data):
        """
        Filter the given data based on the partition data using multiprocessing.

        Parameters
        ----------
        data : list
            The data to filter.
        partition_data : list
            The partition data to match against.

        Returns
        -------
        list
            The filtered data that matches the partition data.
        """

        with multiprocessing.get_context('fork').Pool() as pool:
            arguments = [(x, partition_data) for x in data]
            data = [x for x in pool.starmap(self._validate_filtered_data, arguments) if x]

        return data

    def _validate_filtered_data(self, item, partition_data):
        """
        Validate if the given item matches any partition data.

        Parameters
        ----------
        item : any
            The item to validate.
        partition_data : list
            The partition data to match against.

        Returns
        -------
        any or None
            The item if it matches any partition data, or None if there is no match.
        """

        filtered_data = None

        for image_id in partition_data:
            if image_id in item['image']:
                filtered_data = item

        return filtered_data

    def _load_lines_data(self, file_path):
        """
        Loads the lines data from a file.

        Parameters
        ----------
        file_path : str
            The path to the file containing the lines data.

        Returns
        -------
        list
            A list of lines data.
        """

        lines_data = []
        txt_files = glob.glob(file_path, recursive=True)

        for txt_path in txt_files:
            image_path = os.path.basename(txt_path).replace('.txt', '')
            image_path = os.path.join(self.base_path, 'Images', 'Lines', f"{image_path}.png")

            with open(txt_path, 'r') as file:
                text = file.readline().replace('\n', '').strip()

            lines_data.append({
                'image': image_path,
                'bbox': [],
                'text': text,
                'writer': '1',
            })

        return lines_data

    def _load_paragraphs_data(self, file_path):
        """
        Loads the paragraphs data from a file.

        Parameters
        ----------
        file_path : str
            The path to the file containing the lines data.

        Returns
        -------
        list
            A list of paragraphs data.
        """

        xml_files = glob.glob(file_path, recursive=True)
        paragraphs_data = []

        for xml_path in xml_files:
            image_path = os.path.basename(xml_path).replace('.xml', '')
            image_path = os.path.join(self.base_path, 'Images', 'Pages', f"{image_path}.jpg")

            root = ET.parse(xml_path).getroot()
            namespace = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19'}

            text = []
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')

            for content in root.findall('.//ns:TextLine', namespace):
                text_equiv = content.find('ns:TextEquiv', namespace)
                text_line = text_equiv.find('ns:Unicode', namespace).text.strip()

                if not text_line:
                    continue

                coords = content.find('ns:Coords', namespace)
                points = coords.get('points')
                x_values, y_values = [], []

                if points:
                    for point in points.split(' '):
                        x, y = point.split(',')
                        x_values.append(int(x))
                        y_values.append(int(y))

                else:
                    x_values = [int(point.attrib['x']) for point in coords]
                    y_values = [int(point.attrib['y']) for point in coords]

                min_x = min(min_x, min(x_values))
                max_x = max(max_x, max(x_values))
                min_y = min(min_y, min(y_values))
                max_y = max(max_y, max(y_values))

                text.append(text_line)

            paragraphs_data.append({
                'image': image_path,
                'bbox': [min_x, min_y, max_x - min_x, max_y - min_y],
                'text': '\n'.join(text),
                'writer': '1',
            })

        return paragraphs_data
