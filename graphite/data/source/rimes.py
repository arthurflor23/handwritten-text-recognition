import os
import xml.etree.ElementTree as ET


class Source():
    """
    Represents the Rimes database source.

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
        self.base_path = os.path.join(self.artifact_path, 'rimes')

        self.training_path = os.path.join(self.base_path, 'training_2011')
        self.test_path = os.path.join(self.base_path, 'eval_2011')

        self.training_file_path = os.path.join(self.base_path, 'training_2011.xml')
        self.test_file_path = os.path.join(self.base_path, 'eval_2011_annotated.xml')

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

        if text_level == 'line':
            data['training'] = self._load_lines_data(self.training_file_path, self.training_path)
            data['test'] = self._load_lines_data(self.test_file_path, self.test_path)

        elif text_level == 'paragraph':
            data['training'] = self._load_paragraphs_data(self.training_file_path, self.training_path)
            data['test'] = self._load_paragraphs_data(self.test_file_path, self.test_path)

        return data

    def _load_lines_data(self, file_path, partition_path):
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
        root = ET.parse(file_path).getroot()

        for i, single_page in enumerate(root.findall('SinglePage')):
            file_name = os.path.join(partition_path, single_page.get('FileName'))

            paragraph = single_page.find('Paragraph')
            lines = paragraph.findall('Line')

            for line in lines:
                text = line.get('Value').strip()

                if not text:
                    continue

                x = int(line.get('Left'))
                y = int(line.get('Top'))
                width = int(line.get('Right')) - x
                height = int(line.get('Bottom')) - y

                lines_data.append({
                    'image': file_name,
                    'bbox': [x, y, width, height],
                    'text': text,
                    'writer': str(i + 1),
                })

        return lines_data

    def _load_paragraphs_data(self, file_path, partition_path):
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

        paragraphs_data = []
        root = ET.parse(file_path).getroot()

        for i, single_page in enumerate(root.findall('SinglePage')):
            file_name = os.path.join(partition_path, single_page.get('FileName'))

            paragraph = single_page.find('Paragraph')
            lines = paragraph.findall('Line')

            text = []
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')

            for line in lines:
                text_line = line.get('Value').strip()

                if not text_line:
                    continue

                min_x = min(min_x, int(line.get('Left')))
                max_x = max(max_x, int(line.get('Right')))
                min_y = min(min_y, int(line.get('Top')))
                max_y = max(max_y, int(line.get('Bottom')))

                text.append(text_line)

            paragraphs_data.append({
                'image': file_name,
                'bbox': [min_x, min_y, max_x - min_x, max_y - min_y],
                'text': '\n'.join(text),
                'writer': str(i + 1),
            })

        return paragraphs_data
