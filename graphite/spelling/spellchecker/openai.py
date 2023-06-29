import re
import time
import openai
import tiktoken


class SpellChecker():
    """
    Check and correct spelling errors in texts using the OpenAI API.
    """

    def __init__(self, api_key=None):
        """
        Initializes a new instance of the SpellChecker class.

        Parameters
        ----------
        api_key : str
            The API key to interact with the OpenAI API.
        """

        # https://platform.openai.com/account/api-keys
        openai.api_key = api_key

    def enhance_predictions(self, instruction, predictions, verbose=1):
        """
        Enhances predictions via encoding, batching, API requesting, and updating.

        Parameters
        ----------
        instruction : str
            Instructions for the API.
        predictions : list
            Predicted paths, each as a list of texts.
        verbose : int, optional
            Verbosity mode, by default 1.

        Returns
        -------
        list
            Enhanced predictions in the same structure as input.
        """

        encoding = tiktoken.get_encoding('p50k_edit')
        enhanced_predictions = []

        max_tokens = 2560

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

            if verbose:
                print(f"Enhance top path {i + 1} (batches: {len(batches)})")

            enhanced_data = '\n'.join([self._request_api(instruction, '\n'.join(x)) for x in batches])

            pattern = re.compile(r'<([0-9]+\.[0-9]+)>(.*?)<\/\1>', re.DOTALL)
            matches = pattern.findall(enhanced_data)

            enhanced_texts = [list(sublist) for sublist in top_path]

            if len(matches) == len(enhanced_texts):
                for match in matches:
                    tags = [int(x) for x in match[0].split('.')]
                    enhanced_texts[tags[0]][tags[1]] = match[1].replace('\n', '').strip()

            enhanced_predictions.append(enhanced_texts)

        return enhanced_predictions

    def _request_api(self, instruction, text):
        """
        Makes a request to the OpenAI API.

        Parameters
        ----------
        instruction : str
            The instruction to be followed by the API.
        text : str
            The prompt for the API.

        Returns
        -------
        response : str
            The API response.
        """

        retry_limit = 10
        retry_sleep = 10
        retry_count = 0

        while retry_count < retry_limit:
            try:
                response = openai.Edit.create(engine='text-davinci-edit-001',
                                              instruction=' '.join(instruction.split()),
                                              input=text,
                                              temperature=0,
                                              top_p=1,
                                              n=1)

                response = response.choices[0].text.strip()
                return response

            except Exception as err:
                retry_count += 1
                print(err)
                print(f"Request failed. Retrying... (Attempt {retry_count}/{retry_limit})")

                retry_sleep += 10
                time.sleep(retry_sleep)

        return text
