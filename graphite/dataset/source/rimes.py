import os


class Source():
    """
    Represents the Rimes database source.
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
        self.base_path = os.path.join(self.data_path, 'rimes')

        self.training_path = os.path.join(self.base_path, 'training_2011', 'images')
        self.test_path = os.path.join(self.base_path, 'eval_2011', 'images')

        self.training_file_path = os.path.join(self.base_path, 'training_2011.xml')
        self.test_file_path = os.path.join(self.base_path, 'eval_2011_annotated.xml')
