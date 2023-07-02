import os
import dotenv
import importlib
import numpy as np


class Spelling():
    """
    Check and correct spelling errors in texts data.
    """

    def __init__(self,
                 spell_checker=None,
                 api_key=None,
                 env_key=None,
                 dotenv_file='.env'):
        """
        Initializes a new instance of the Spelling class.

        Parameters
        ----------
        spell_checker : str, optional
            The spell checker name. Default is None.
        api_key : str, optional
            The API key to interact with spell checker. Default is None.
        env_key : str, optional
            The key to access the environment variable which holds the API key. Default is None.
        dotenv_file : str, optional
            The file name of the environment file. Default is '.env'.
        """

        self.spell_checker = spell_checker
        self.api_key = api_key
        self.env_key = env_key
        self.dotenv_path = dotenv_file

        if self.env_key is not None:
            dotenv.load_dotenv(self.dotenv_path)
            self.api_key = os.environ[self.env_key]

        self._spell_checker = None

        if spell_checker:
            self._spell_checker = self._import_spell_checker(self.spell_checker)
            self._spell_checker = self._spell_checker(self.api_key)

    def __repr__(self):
        """
        Returns a string representation of the object with useful information.

        Returns
        -------
        str
            The string representation of the object.
        """

        info = f"""
            Spelling Configuration\n
            Spell Checker           {self.spell_checker}
            API Key                 {'*****' if self.api_key else '-'}
            Environment Key         {self.env_key or '-'}
            Environment Path        {self.dotenv_path or '-'}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()]).strip()

        return info

    def to_dict(self):
        """
        Convert the class object attributes to a dictionary.

        Returns
        -------
        dict
            A dictionary with the class attributes.
        """

        attributes = {
            'spell_checker': self.spell_checker,
            'env_key': self.env_key,
            'dotenv_path': self.dotenv_path,
        }

        return attributes

    def enhance(self, predictions, instruction=None, verbose=1):
        """
        Enhances predictions by correcting spelling errors.

        Parameters
        ----------
        predictions : list
            The predictions to be enhanced.
        instruction : str, optional
            The instruction to be followed by the API.
        verbose : int, optional
            Verbosity mode, by default 1.

        Returns
        -------
        enhanced : list
            The enhanced predictions.
        """

        if instruction is None:
            instruction = """
                Correct spelling errors, including accents.
                The following texts contain errors from a handwriting recognition model.
                Preserve slang, historical terms, and grammar. Make only confident changes.
            """

        if self.spell_checker is not None:
            predictions = self._spell_checker.enhance_predictions(instruction, predictions, verbose)
            predictions = np.array(predictions, dtype=object)

        return predictions

    def _import_spell_checker(self, spell_checker):

        module_name = importlib.util.resolve_name(f".spellchecker.{spell_checker}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "spelling file must be created"

        module = importlib.import_module(module_name, __package__)

        class_name = 'SpellChecker'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        spell_checker = getattr(module, class_name)

        return spell_checker
