import os
import glob
import multiprocessing
import xml.etree.ElementTree as ET


class Source():
    """
    Represents the Bentham database source.
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

    def fetch_data(self, level):
        """
        Retrieves the data for training, validation, and testing.

        Parameters
        ----------
        level : str
            The granularity of the data to be retrieved.

        Returns
        -------
        tuple
            A tuple containing lists of training, validation, and test data.
        """

        # Load the partition data for training, validation, and testing
        training_partition_data = self._load_partition_data(self.training_file_path)
        validation_partition_data = self._load_partition_data(self.validation_file_path)
        test_partition_data = self._load_partition_data(self.test_file_path)

        training_data, validation_data, test_data = [], [], []

        if level == 'line':
            # Load the lines data from the files
            lines_data = self._load_lines_data(self.lines_file_path)

            # Filter the lines data based on the partition data
            training_data = self._filter_data(lines_data, training_partition_data)
            validation_data = self._filter_data(lines_data, validation_partition_data)
            test_data = self._filter_data(lines_data, test_partition_data)

        elif level == 'paragraph':
            # Load the paragraphs data from the files
            paragraphs_data = self._load_paragraphs_data(self.paragraphs_file_path)

            # Filter the paragraphs data based on the partition data
            training_data = self._filter_data(paragraphs_data, training_partition_data)
            validation_data = self._filter_data(paragraphs_data, validation_partition_data)
            test_data = self._filter_data(paragraphs_data, test_partition_data)

        return training_data, validation_data, test_data

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
            if image_id in item[0]:
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

        txt_files = glob.glob(file_path, recursive=True)
        line_data = []

        for txt_path in txt_files:
            image_path = os.path.basename(txt_path).replace('.txt', '')
            image_path = os.path.join(self.base_path, 'Images', 'Lines', f"{image_path}.png")

            with open(txt_path, 'r') as file:
                label = file.readline().replace('\n', '').strip()

            line_data.append([image_path, [], label])

        return line_data

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
        paragraph_data = []

        for xml_path in xml_files:
            image_path = os.path.basename(xml_path).replace('.xml', '')
            image_path = os.path.join(self.base_path, 'Images', 'Pages', f"{image_path}.jpg")

            root = ET.parse(xml_path).getroot()
            namespace = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19'}

            label = []
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')

            for text_line in root.findall('.//ns:TextLine', namespace):
                text_equiv = text_line.find('ns:TextEquiv', namespace)
                line_label = text_equiv.find('ns:Unicode', namespace).text.strip()

                if not line_label:
                    continue

                coords = text_line.find('ns:Coords', namespace)
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

                label.append(line_label)

            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
            paragraph_data.append([image_path, bbox, label])

        return paragraph_data
