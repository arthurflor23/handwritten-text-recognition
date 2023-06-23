import time
import openai


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

    def enhance_text(self, text, instruction=None):
        """
        Makes a request to the OpenAI API.

        Parameters
        ----------
        text : str
            The prompt for the API.
        instruction : str, optional
            The instruction to be followed by the API.

        Returns
        -------
        response : str
            The API response.
        """

        if instruction is None:
            instruction = """
                Correct any spelling errors, including accents.
                Preserve slang, historical terms, and grammar.
                Treat the content of each tag as separate.
                Make only confident changes.
            """

        retry_limit = 10
        retry_sleep = 10
        retry_count = 0

        while retry_count < retry_limit:
            try:
                response = openai.Edit.create(engine='text-davinci-edit-001',
                                              instruction=' '.join(instruction.split()),
                                              input=text or '',
                                              temperature=0,
                                              top_p=1,
                                              n=1)

                response = response.choices[0].text.strip()
                return response

            except Exception as err:
                retry_count += 1
                print(err)
                print(f"Request failed. Retrying... (Attempt {retry_count}/{retry_limit})")

                retry_sleep += 30
                time.sleep(retry_sleep)

        return text
