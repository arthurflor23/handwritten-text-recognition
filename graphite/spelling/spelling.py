import os
import re
import copy
import dotenv
import importlib
import concurrent
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

        if spell_checker:
            self._spell_checker = self._import_spell_checker(self.spell_checker)
            self._spell_checker = self._spell_checker(self.api_key)

    def __repr__(self):
        """
        Returns a JSON-formatted string representation of the object.

        Returns
        -------
        str
            A JSON-formatted string containing the object's attributes.
        """

        attributes = {
            'spell_checker': self.spell_checker,
            'env_key': self.env_key,
            'dotenv_path': self.dotenv_path,
        }

        return attributes

    def __str__(self):
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
            Environment Key         {self.env_key or '-'}
            Environment Path        {self.dotenv_path or '-'}
        """

        info = '\n'.join([x.strip() for x in info.splitlines()])

        return info

    def enhance(self, predictions, instruction=None):
        """
        Enhances predictions by correcting spelling errors.

        Parameters
        ----------
        predictions : list
            The predictions to be enhanced.
        instruction : str, optional
            The instruction to be followed by the API.

        Returns
        -------
        enhanced : list
            The enhanced predictions.
        """

        if not self.spell_checker:
            return predictions

        enhanced = []

        for texts in predictions:
            tokens_length = 0
            batches = [[]]

            for i, text in enumerate(texts):
                for j, line in enumerate(text):
                    pp_text = f'<{i}.{j}>{line}</{i}.{j}>'
                    pp_text_tokens_length = len(pp_text.split())

                    if tokens_length + pp_text_tokens_length > 900:
                        batches.append([])
                        tokens_length = 0
                    else:
                        tokens_length += pp_text_tokens_length

                    batches[-1].append(pp_text)

            print(f"Total batches: {len(batches)}")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._spell_checker.enhance_text,
                                           '\n'.join(x), instruction) for x in batches]
                enhanced_batches = [future.result() for future in futures]

            enhanced_texts = copy.deepcopy(texts)
            pattern = re.compile(r'<([0-9]+\.[0-9]+)>(.*?)<\/\1>', re.DOTALL)

            for enhanced_batch in enhanced_batches:
                matches = pattern.findall(enhanced_batch)

                for match in matches:
                    tags = [int(x) for x in match[0].split('.')]
                    enhanced_texts[tags[0]][tags[1]] = match[1].replace('\n', '').strip()

            enhanced.append(enhanced_texts)

        enhanced = np.array(enhanced, dtype=object)

        return enhanced

    def _import_spell_checker(self, spell_checker):

        module_name = importlib.util.resolve_name(f".spellchecker.{spell_checker}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "spelling file must be created"

        module = importlib.import_module(module_name, __package__)

        class_name = 'SpellChecker'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        spell_checker = getattr(module, class_name)

        return spell_checker
