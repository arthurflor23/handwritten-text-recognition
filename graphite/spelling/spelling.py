import os
import re
import copy
import dotenv
import tiktoken
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

        info = '\n'.join([x.strip() for x in info.splitlines()])

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

        if instruction is None:
            instruction = """
                Correct spelling errors, including accents.
                The following texts contain errors from a handwriting recognition model.
                Preserve slang, historical terms, and grammar. Make only confident changes.
            """

        max_tokens = 2560

        encoding = tiktoken.get_encoding('gpt2')
        enhanced_top_paths = []

        for i, top_path in enumerate(predictions):
            tokens_length = 0
            batches = [[]]

            for j, text in enumerate(top_path):
                for u, line in enumerate(text):
                    pp_text = f'<{j}.{u}> {line} </{j}.{u}>'
                    pp_text_tokens_length = len(encoding.encode(pp_text))

                    if tokens_length + pp_text_tokens_length > max_tokens:
                        batches.append([])
                        tokens_length = 0
                    else:
                        tokens_length += pp_text_tokens_length

                    batches[-1].append(pp_text)

            print(f"Enhance top path {i + 1} (batches: {len(batches)})")

            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self._spell_checker.enhance_text,
                                           '\n'.join(x), instruction) for x in batches]
                enhanced_data = '\n'.join([future.result() for future in futures])

            pattern = re.compile(r'<([0-9]+\.[0-9]+)>(.*?)<\/\1>', re.DOTALL)
            matches = pattern.findall(enhanced_data)

            enhanced_texts = copy.deepcopy(top_path)

            if len(matches) == len(enhanced_texts):
                for match in matches:
                    tags = [int(x) for x in match[0].split('.')]
                    enhanced_texts[tags[0]][tags[1]] = match[1].replace('\n', '').strip()

            enhanced_top_paths.append(enhanced_texts)

        enhanced_top_paths = np.array(enhanced_top_paths, dtype=object)

        return enhanced_top_paths

    def _import_spell_checker(self, spell_checker):

        module_name = importlib.util.resolve_name(f".spellchecker.{spell_checker}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "spelling file must be created"

        module = importlib.import_module(module_name, __package__)

        class_name = 'SpellChecker'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        spell_checker = getattr(module, class_name)

        return spell_checker
