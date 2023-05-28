import os
import glob
import multiprocessing


class Source():
    """
    Represents the IAM database source.
    """

    def __init__(self, data_path):
        """
        Initializes a new instance of the Source class.

        Parameters
        ----------
        data_path : str
            The path to the data.

        Returns
        -------
        None
        """

        self.data_path = data_path
        self.base_path = os.path.join(self.data_path, 'iam')

        self.partition_path = os.path.join(self.base_path, 'largeWriterIndependentTextLineRecognitionTask')
        self.training_file_path = os.path.join(self.partition_path, 'trainset.txt')
        self.validation_file_path = os.path.join(self.partition_path, 'validationset1.txt')
        self.test_file_path = os.path.join(self.partition_path, 'testset.txt')

        self.transcription_path = os.path.join(self.base_path, 'ascii')
        self.words_file_path = os.path.join(self.transcription_path, 'words.txt')
        self.lines_file_path = os.path.join(self.transcription_path, 'lines.txt')

    def get_word_data(self):
        """
        Retrieves the word data for training, validation, and testing.

        Returns
        -------
        tuple
            A tuple containing lists of training, validation, and test words data.
        """

        # Load the partition data for training, validation, and testing
        training_data = self._load_partition_data(self.training_file_path)
        validation_data = self._load_partition_data(self.validation_file_path)
        test_data = self._load_partition_data(self.test_file_path)

        # Load the words data from the file
        words_data = self._load_words_data(self.words_file_path)

        # Filter the words data based on the partition data
        training_words = self._filter_data(words_data, training_data)
        validation_words = self._filter_data(words_data, validation_data)
        test_words = self._filter_data(words_data, test_data)

        return training_words, validation_words, test_words

    def get_line_data(self):
        """
        Retrieves the line data for training, validation, and testing.

        Returns
        -------
        tuple
            A tuple containing lists of training, validation, and test lines data.
        """

        # Load the partition data for training, validation, and testing
        training_partition_data = self._load_partition_data(self.training_file_path)
        validation_partition_data = self._load_partition_data(self.validation_file_path)
        test_partition_data = self._load_partition_data(self.test_file_path)

        # Load the lines data from the file
        lines_data = self._load_lines_data(self.lines_file_path)

        # Filter the lines data based on the partition data
        training_lines = self._filter_data(lines_data, training_partition_data)
        validation_lines = self._filter_data(lines_data, validation_partition_data)
        test_lines = self._filter_data(lines_data, test_partition_data)

        return training_lines, validation_lines, test_lines

    def get_paragraph_data(self):
        """
        Retrieves the paragraph data for training, validation, and testing.

        Returns
        -------
        tuple
            A tuple containing lists of training, validation, and test paragraphs data.
        """

        # Load the partition data for training, validation, and testing
        training_partition_data = self._load_partition_data(self.training_file_path)
        validation_partition_data = self._load_partition_data(self.validation_file_path)
        test_partition_data = self._load_partition_data(self.test_file_path)

        # Load the lines data from the file, including bounding box information
        lines_data = self._load_lines_data(self.lines_file_path, bbox_info=True)
        # Construct paragraphs data from lines data, without bounding box information
        paragraphs_data = self._load_paragraphs_data(lines_data, bbox_info=True)

        # Filter the paragraphs data based on the partition data
        training_paragraphs = self._filter_data(paragraphs_data, training_partition_data, form_group=True)
        validation_paragraphs = self._filter_data(paragraphs_data, validation_partition_data, form_group=True)
        test_paragraphs = self._filter_data(paragraphs_data, test_partition_data, form_group=True)

        return training_paragraphs, validation_paragraphs, test_paragraphs

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

    def _filter_data(self, data, partition_data, form_group=False):
        """
        Filter the given data based on the partition data using multiprocessing.

        Parameters
        ----------
        data : list
            The data to filter.
        partition_data : list
            The partition data to match against.
        form_group : bool, optional
            Indicates whether to group the image ID, by default False.

        Returns
        -------
        list
            The filtered data that matches the partition data.
        """

        with multiprocessing.get_context('fork').Pool() as pool:
            arguments = [(x, partition_data, form_group) for x in data]
            data = [x for x in pool.starmap(self._validate_filtered_data, arguments) if x]

        return data

    def _validate_filtered_data(self, item, partition_data, form_group=False):
        """
        Validate if the given item matches any partition data.

        Parameters
        ----------
        item : any
            The item to validate.
        partition_data : list
            The partition data to match against.
        form_group : bool, optional
            Indicates whether to group the image ID, by default False.

        Returns
        -------
        any or None
            The item if it matches any partition data, or None if there is no match.
        """

        filtered_data = None

        for image_id in partition_data:
            if form_group:
                image_id = '-'.join(image_id.split('-')[:2])

            if image_id in item[0]:
                filtered_data = item

        return filtered_data

    def _load_words_data(self, file_path, bbox_info=False):
        """
        Loads the words data from a file.

        Parameters
        ----------
        file_path : str
            The path to the file containing the words data.
        bbox_info : bool, optional
            Determines whether to include bounding box information in the words data.

        Returns
        -------
        list
            A list of words data.
        """

        with open(file_path, 'r') as file:
            rows = file.readlines()

        words_data = []

        for row in rows:
            if row.startswith('#'):
                continue

            parts = row.split()
            word_ids = parts[0].split('-')
            word_path = os.path.join('words', word_ids[0], '-'.join(word_ids[:2]))
            word_file_name = f"{parts[0]}.png"

            image_path = os.path.join(self.base_path, word_path, word_file_name)
            bbox = [int(x) for x in parts[4:8]] if bbox_info else []
            label = ' '.join(parts[8:]).replace('|', ' ')

            # Construct the word data entry with image path, bounding box, and label
            words_data.append([image_path, bbox, label])

        return words_data

    def _load_lines_data(self, file_path, bbox_info=False):
        """
        Loads the lines data from a file.

        Parameters
        ----------
        file_path : str
            The path to the file containing the lines data.
        bbox_info : bool, optional
            Determines whether to include bounding box information in the lines data.

        Returns
        -------
        list
            A list of lines data.
        """

        with open(file_path, 'r') as file:
            rows = file.readlines()

        lines_data = []

        for row in rows:
            if row.startswith('#'):
                continue

            parts = row.split()
            line_ids = parts[0].split('-')
            line_path = os.path.join('lines', line_ids[0], '-'.join(line_ids[:2]))
            line_file_name = f"{parts[0]}.png"

            image_path = os.path.join(self.base_path, line_path, line_file_name)
            bbox = [int(x) for x in parts[4:8]] if bbox_info else []
            label = ' '.join(parts[8:]).replace('|', ' ')

            # Construct the line data entry with image path, bounding box, and label
            lines_data.append([image_path, bbox, label])

        return lines_data

    def _load_paragraphs_data(self, lines_data, bbox_info=False):
        """
        Loads the paragraphs data from a file.

        Parameters
        ----------
        lines_data : str
            The lines data as base content to paragraph data construction.
        bbox_info : bool, optional
            Determines whether to include bounding box information in the paragraphs data.

        Returns
        -------
        list
            A list of paragraphs data.
        """

        paragraphs_data = {}

        for line in lines_data:
            parts = os.path.basename(line[0]).split('-')
            pattern = os.path.join(self.base_path, "forms**", f"{'-'.join(parts[:2])}.png")
            file_paths = glob.glob(pattern, recursive=True)

            if not file_paths:
                continue

            file_path = file_paths[0]

            if file_path in paragraphs_data:
                images_path, bbox, label = paragraphs_data[file_path]

                # Update the bounding box by expanding it to include the line
                bbox[0] = min(bbox[0], line[1][0])
                bbox[1] = min(bbox[1], line[1][1])
                bbox[2] = max(bbox[2], line[1][2] + line[1][0])
                bbox[3] = max(bbox[3], line[1][3] + line[1][1])

                # Append the line's label to the existing paragraph's label
                label += f"\n{line[2]}"

                paragraphs_data[file_path] = [images_path, bbox, label]

            else:
                bbox = [line[1][0], line[1][1], line[1][2] + line[1][0], line[1][3] + line[1][1]]
                paragraphs_data[file_path] = [file_path, bbox, line[2]]

        paragraphs_data = list(paragraphs_data.values())

        for i, paragraph in enumerate(paragraphs_data):
            if bbox_info:
                bbox = paragraph[1]

                # Update the bounding box to be a list of coordinates
                bbox[0] = paragraph[1][0]
                bbox[1] = paragraph[1][1]
                bbox[2] = abs(paragraph[1][2] - bbox[0])
                bbox[3] = abs(paragraph[1][3] - bbox[1])

                paragraphs_data[i][1] = bbox
            else:
                # If bbox_info is False, set an empty list for the bounding box
                paragraphs_data[i][1] = []

        return paragraphs_data
