import os
import xml.etree.ElementTree as ET


class Source():
    """
    Represents the Rimes database source.
    """

    def __init__(self, artifact_path):
        """
        Initializes a new instance of the Source class.

        Parameters
        ----------
        artifact_path : str
            The path to the data.

        Returns
        -------
        None
        """

        self.artifact_path = artifact_path
        self.base_path = os.path.join(self.artifact_path, 'rimes')

        self.training_path = os.path.join(self.base_path, 'training_2011')
        self.test_path = os.path.join(self.base_path, 'eval_2011')

        self.training_file_path = os.path.join(self.base_path, 'training_2011.xml')
        self.test_file_path = os.path.join(self.base_path, 'eval_2011_annotated.xml')

    def fetch_data(self, level):
        """
        Retrieves the data for training, validation, and testing.

        Parameters
        ----------
        level : str
            The granularity level of the data to be retrieved.

        Returns
        -------
        tuple
            A tuple containing lists of training, validation, and test data.
        """

        if level == 'line':
            # Load the lines data from the files
            training_data = self._load_lines_data(self.training_file_path, self.training_path)
            test_data = self._load_lines_data(self.test_file_path, self.test_path)

        elif level == 'paragraph':
            # Load the paragraphs data from the files
            training_data = self._load_paragraphs_data(self.training_file_path, self.training_path)
            test_data = self._load_paragraphs_data(self.test_file_path, self.test_path)

        return training_data, [], test_data

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

        line_data = []

        # Parse the XML file
        root = ET.parse(file_path).getroot()

        # Iterate over each SinglePage elements
        for single_page in root.findall('SinglePage'):
            # Get the FileName attribute and construct the full file path
            file_name = os.path.join(partition_path, single_page.get('FileName'))

            # Find the Paragraph element within SinglePage
            paragraph = single_page.find('Paragraph')
            # Find all Line elements within Paragraph
            lines = paragraph.findall('Line')

            # Iterate over each Line element
            for line in lines:
                # Extract the attributes of each Line element
                label = line.get('Value').strip()

                if not label:
                    continue

                x = int(line.get('Left'))
                y = int(line.get('Top'))
                width = int(line.get('Right')) - x
                height = int(line.get('Bottom')) - y

                # Append the line data to lines_data
                line_data.append([file_name, [x, y, width, height], label])

        return line_data

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

        paragraph_data = []

        # Parse the XML file
        root = ET.parse(file_path).getroot()

        # Iterate over each SinglePage elements
        for single_page in root.findall('SinglePage'):
            # Get the FileName attribute and construct the full file path
            file_name = os.path.join(partition_path, single_page.get('FileName'))

            # Find the Paragraph element within SinglePage
            paragraph = single_page.find('Paragraph')
            # Find all Line elements within Paragraph
            lines = paragraph.findall('Line')

            label = []
            min_x, max_x = float('inf'), float('-inf')
            min_y, max_y = float('inf'), float('-inf')

            # Iterate over each Line element
            for line in lines:
                # Extract the attributes of each Line element
                line_label = line.get('Value').strip()

                if not line_label:
                    continue

                # Get the minimum and maximum coords x and y
                min_x = min(min_x, int(line.get('Left')))
                max_x = max(max_x, int(line.get('Right')))
                min_y = min(min_y, int(line.get('Top')))
                max_y = max(max_y, int(line.get('Bottom')))

                # Get label
                label.append(line_label)

            # Concatenate the Value attribute of Line elements
            bbox = [min_x, min_y, max_x - min_x, max_y - min_y]

            # Append the file name, bbox, and label
            paragraph_data.append([file_name, bbox, label])

        return paragraph_data
