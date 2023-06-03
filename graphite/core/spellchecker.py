import os
import re
import time
import dotenv
import openai
import concurrent


class SpellChecker():

    def __init__(self, api_key=None, env_key=None, dotenv_file='.env'):

        self.api_key = api_key
        self.env_key = env_key

        self.base_path = os.path.join(os.path.dirname(__file__), '..')
        self.dotenv_path = os.path.join(self.base_path, dotenv_file)

        if self.env_key is not None:
            dotenv.load_dotenv(self.dotenv_path)
            self.api_key = os.environ[self.env_key]

        # https://platform.openai.com/account/api-keys
        openai.api_key = self.api_key

    def enhance_texts(self, texts, instruction=None):

        tokens_length = 0
        batches = [[]]

        for i, text in enumerate(texts):
            for j, line in enumerate(text):
                pp_text = f"<{i}.{j}>{line}</{i}.{j}>"
                pp_text_tokens_length = len(pp_text.split())

                if tokens_length + pp_text_tokens_length > 1024:
                    batches.append([])
                    tokens_length = 0
                else:
                    tokens_length += pp_text_tokens_length

                batches[-1].append(pp_text)

        print(f"Total batches: {len(batches)}")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(self._request_api, instruction, "\n".join(x)) for x in batches]
            enhanced_batches = [future.result() for future in futures]

        pattern = re.compile(r'<([0-9]+\.[0-9]+)>(.*?)<\/\1>', re.DOTALL)
        enhanced_texts = texts.copy()

        for enhanced_batch in enhanced_batches:
            matches = pattern.findall(enhanced_batch)

            for match in matches:
                tag = match[0].split(".")
                enhanced_texts[int(tag[0])][int(tag[1])] = match[1].replace("\n", "").strip()

        # remove this
        for i in range(len(enhanced_texts)):
            print(texts[i])
            print(enhanced_texts[i])
            print()

        return enhanced_texts

    def _request_api(self, instruction=None, prompt=None):

        if instruction is None:
            instruction = """Fix all spelling mistakes (including accents and contractions)."""

        retry_limit = 10
        retry_sleep = 10
        retry_count = 0

        while retry_count < retry_limit:
            try:
                response = openai.Edit.create(engine="text-davinci-edit-001",
                                              instruction=' '.join(instruction.split()),
                                              input=prompt or '',
                                              temperature=0,
                                              top_p=1,
                                              n=16)

                response = response.choices[0].text.strip()
                return response

            except Exception as err:
                retry_count += 1
                print(err)
                print(f"Request failed. Retrying... (Attempt {retry_count}/{retry_limit})")

                retry_sleep += 5
                time.sleep(retry_sleep)

        return prompt
