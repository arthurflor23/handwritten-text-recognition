import os
import re
import copy
import dotenv
import importlib
import concurrent


class Spelling():
    """
    Check and correct spelling errors in texts data.
    """

    def __init__(self,
                 spell_checker,
                 api_key=None,
                 env_key=None,
                 dotenv_file='.env'):
        """
        Initializes a new instance of the Spelling class.

        Parameters
        ----------
        spell_checker : str
            The spell checker name.
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

        self.base_path = os.path.join(os.path.dirname(__file__), '..')
        self.dotenv_path = os.path.join(self.base_path, dotenv_file)

        if self.env_key is not None:
            dotenv.load_dotenv(self.dotenv_path)
            self.api_key = os.environ[self.env_key]

        self._spell_checker = self._import_spell_checker(self.spell_checker)
        self._spell_checker = self._spell_checker(self.api_key)

    def enhance(self, text_data, instruction=None):
        """
        Enhances texts by correcting spelling errors.

        Parameters
        ----------
        texts : list
            The texts to be enhanced.
        instruction : str, optional
            The instruction to be followed by the API.

        Returns
        -------
        enhanced_texts : list
            The enhanced texts.
        """

        tokens_length = 0
        batches = [[]]

        for i, text in enumerate(text_data):
            for j, line in enumerate(text):
                pp_text = f'<{i}.{j}>{line}</{i}.{j}>'
                pp_text_tokens_length = len(pp_text.split())

                if tokens_length + pp_text_tokens_length > 1024:
                    batches.append([])
                    tokens_length = 0
                else:
                    tokens_length += pp_text_tokens_length

                batches[-1].append(pp_text)

        print(f"Total batches: {len(batches)}")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._spell_checker.enhance, '\n'.join(x), instruction) for x in batches]
            enhanced_batches = [future.result() for future in futures]

        enhanced_texts = copy.deepcopy(text_data)
        pattern = re.compile(r'<([0-9]+\.[0-9]+)>(.*?)<\/\1>', re.DOTALL)

        for enhanced_batch in enhanced_batches:
            matches = pattern.findall(enhanced_batch)

            for match in matches:
                tags = [int(x) for x in match[0].split('.')]
                enhanced_texts[tags[0]][tags[1]] = match[1].replace('\n', '').strip()

        return enhanced_texts

    def _import_spell_checker(self, spell_checker):

        module_name = importlib.util.resolve_name(f".spellchecker.{spell_checker}", __package__)
        module_spec = importlib.util.find_spec(module_name)
        assert module_spec is not None, "spelling file must be created"

        module = importlib.import_module(module_name, __package__)

        class_name = 'SpellChecker'
        assert hasattr(module, class_name), f"`{class_name}` class must be created"

        spell_checker = getattr(module, class_name)

        return spell_checker
