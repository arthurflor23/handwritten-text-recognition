import os
import re
import time
import string
import openai
import numpy as np
import tensorflow as tf


class SpellingModel():
    """
    A model for correcting spelling errors in text using OpenAI's GPT model.
    """

    def __init__(self, env_file='.env', env_key='OPENAI_API_KEY'):
        """
        Initializes the spelling model.

        Parameters
        ----------
        env_file : str, optional
            Path to the environment file.
        env_key : str, optional
            Environment variable key for API key.
        """

        self.env_file = env_file
        self.env_key = env_key

        self.model = 'gpt-4o-mini'
        self.max_tokens = 8192

        self.instruction = ('Correct only obvious spelling mistakes in words within tags. '
                            'Keep the number of tags the same. '
                            'Do not add extra text or change correct text. '
                            'Maintain the unique and historical style of the text.')

        openai.api_key = self._get_api_key()

    def _get_api_key(self):
        """
        Retrieves the API key from the environment file or environment variables.

        Returns
        -------
        str
            Retrieved API key.
        """

        if os.path.isfile(self.env_file):
            with open(self.env_file, 'r') as file:
                for line in file:
                    if line.startswith(f"{self.env_key}="):
                        return line.split('=', 1)[1].strip()

        return os.getenv(self.env_key)

    def _encode_batch(self, batch):
        """
        Encodes a batch of text.

        Parameters
        ----------
        batch : list
            Batch of text data.

        Returns
        -------
        list
            Encoded text data.
        """

        tokens_length = 0
        encoded = [[]]

        for i, data in enumerate(batch):
            for j, top_path in enumerate(data):
                for u, text in enumerate(top_path.split('\n')):
                    pp_text = f'<{i}.{j}.{u}> {text} </{i}.{j}.{u}>'

                    pp_text_tokens = re.sub(f'([{re.escape(string.punctuation)}])', r' \1 ', pp_text).split()
                    pp_text_tokens_length = len(pp_text_tokens)

                    if tokens_length + pp_text_tokens_length >= (self.max_tokens // 2):
                        encoded.append([])
                        tokens_length = 0

                    tokens_length += pp_text_tokens_length
                    encoded[-1].append(pp_text)

        return encoded

    def _decode_batch(self, batch, encoded_batch, corrected_encoded_batch):
        """
        Decodes a processed batch of text.

        Parameters
        ----------
        batch : list
            Batch of text data.
        encoded_batch : list
            Encoded batch of text data for fallback.
        corrected_encoded_batch : list
            Processed and corrected text data.

        Returns
        -------
        list
            Decoded and corrected text data.
        """

        if len(corrected_encoded_batch) != len(encoded_batch):
            return batch

        if not isinstance(batch, list):
            batch = batch.tolist()

        pattern = re.compile(r'<([0-9]+\.[0-9]+\.[0-9]+)>(.*?)<\/\1>', re.DOTALL)

        for i, (corrected, fallback) in enumerate(zip(corrected_encoded_batch, encoded_batch)):
            corrected_encoded_batch_matches = pattern.findall(''.join(corrected))
            fallback_encoded_matches = pattern.findall(''.join(fallback))

            if len(corrected_encoded_batch_matches) != len(fallback_encoded_matches):
                corrected_encoded_batch[i] = fallback

            for match in corrected_encoded_batch_matches:
                tags = tuple(map(int, match[0].split('.')))
                batch_item = batch[tags[0]][tags[1]]

                if isinstance(batch_item, str):
                    batch[tags[0]][tags[1]] = []

                batch[tags[0]][tags[1]].append(match[1].strip())

        for i, data in enumerate(batch):
            for j, item in enumerate(data):
                if isinstance(item, list):
                    batch[i][j] = '\n'.join(item)

        return batch

    def _request_api(self, batch):
        """
        Sends a request to the OpenAI API and handles retries.

        Parameters
        ----------
        batch : list
            Batch of text data to send.

        Returns
        -------
        list
            Processed text responses.
        """

        messages = [
            {'role': 'system', 'content': self.instruction},
            {'role': 'user', 'content': '\n\n'.join(batch)},
        ]

        retry_limit = 10
        retry_sleep = 10

        for attempt in range(1, retry_limit + 1):
            try:
                response = openai.chat.completions.create(model=self.model,
                                                          messages=messages,
                                                          temperature=0,
                                                          top_p=1.0,
                                                          n=1,
                                                          seed=0)

                return response.choices[0].message.content.strip().split('\n')

            except Exception as err:
                print(f"OpenAI message error: {err}")

                if attempt <= retry_limit:
                    print(f"Request failed. Retrying... (attempt {attempt}/{retry_limit})")
                    time.sleep(retry_sleep)
                    retry_sleep += 10

        return batch

    def predict(self, x, steps, verbose=1):
        """
        Predicts the corrections for the given data of texts.

        Parameters
        ----------
        x : list
            Data of texts.
        steps : int
            Number of steps for processing.
        verbose : int, optional
            Verbosity mode.

        Returns
        -------
        np.ndarray
            Array of corrected texts.
        """

        if not openai.api_key:
            return x

        progbar = tf.keras.utils.Progbar(target=steps, unit_name='spelling', verbose=verbose)

        corrections = []
        batch_index = 0
        batch_size = int(np.ceil(len(x) / steps))

        for step in range(steps):
            progbar.update(step)

            batch = x[batch_index:batch_index + batch_size]
            batch_index += batch_size

            encoded_batch = self._encode_batch(batch)
            corrected_encoded_batch_batch = [self._request_api(item) for item in encoded_batch]
            corrected_batch = self._decode_batch(batch, encoded_batch, corrected_encoded_batch_batch)

            corrections.extend(corrected_batch)
            progbar.update(step + 1)

        corrections = np.array(corrections, dtype=object)

        return corrections
