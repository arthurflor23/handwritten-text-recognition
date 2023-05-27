import os


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

        self.transcription_path = os.path.join(self.base_path, 'ascii')
        self.partition_path = os.path.join(self.base_path, 'largeWriterIndependentTextLineRecognitionTask')

        self.training_file_path = os.path.join(self.partition_path, 'trainset.txt')
        self.validation_file_path = os.path.join(self.partition_path, 'validationset1.txt')
        self.test_file_path = os.path.join(self.partition_path, 'testset.txt')

        self.words_file_path = os.path.join(self.transcription_path, 'words.txt')
        self.lines_file_path = os.path.join(self.transcription_path, 'lines.txt')

    def get_line_data(self):
        factor = 1

        images = [os.path.join(self.data_path, 'sample.png'), os.path.join(self.data_path, 'sample.png')] * factor
        # labels = [". . . this is #80 calling . . . '", ". . . this is #80 calling . . . '"] * factor
        # labels = [[". . . this is #80 calling . . . '"], [". . . this is #80 calling . . . '"]] * factor
        labels = ["Mon habitation étant assurée par votre\nsociété,\tj&apos;aimerais recevoir une",
                  ". . . this is #80 calling . . . '"] * factor
        # cropping = [[0, 0, 0, 0]]
        cropping = []

        training = [images, None, labels]

        validation = []
        test = []

        # return images
        # return [images, labels]
        # return [images, cropping, labels]
        # return training
        return training, validation
