
class Tokenizer():
    """
    A tokenizer class for encoding and decoding text and writer IDs.

    This class provides functionality to convert text into encoded representations and
        vice versa, as well as handling encoding and decoding of writer IDs.
    It maintains statistical metadata about the text processed.
    """

    def __init__(self):
        """
        Initialize the Tokenizer instance.
        """

        self.pad_tk = '¶'
        self.sos_tk = '◖'
        self.eos_tk = '◗'
        self.unk_tk = '◬'

        self.lexical_shape = []
        self.writers = [self.unk_tk]

        self.words = []
        self.chars = [self.pad_tk, self.sos_tk, self.eos_tk, self.unk_tk]

        self._initialize_metadata()

    def __repr__(self):
        """
        Provides a formatted string with useful information.

        Returns
        -------
        str
            Formatted string with useful information.
        """

        info = '=================================================='
        info += f'\n{self.__class__.__name__.center(50)}'
        info += '\n=================================================='
        info += f"\n{'writers':<{25}}: {len(self.writers) - 1:,}"
        info += f"\n{'words':<{25}}: {len(self.words):,}"
        info += f"\n{'chars':<{25}}: {len(self.chars) - 4:,}"
        info += f"\n{'lexical_shape':<{25}}: {self.lexical_shape}"
        info += "\n--------------------------------------------------"

        chars = ''.join(self.chars)
        chunks = [chars[i:i+20] for i in range(0, len(chars), 20)]

        info += f"\n{'charset':<{25}}: {chunks[0]}"
        for chunk in chunks[1:]:
            info += f"\n{'':<26}  {chunk}"

        info += "\n--------------------------------------------------"

        for key, value in self.metadata.items():
            info += f"\n{key:26}: {value:,}"

        return info

    def _initialize_metadata(self):
        """
        Initializes metadata dictionary with statistical keys for text analysis.
        """

        self.stats_keys = [
            'paragraphs_per_page',
            'lines_per_page',
            'lines_per_paragraph',
            'words_per_page',
            'words_per_paragraph',
            'words_per_line',
            'chars_per_page',
            'chars_per_paragraph',
            'chars_per_line',
            'chars_per_word',
        ]

        self.metadata = {}
        self.metadata.update({f'min_{key}': float('inf') for key in self.stats_keys})
        self.metadata.update({f'max_{key}': 0 for key in self.stats_keys})
        self.metadata.update({f'avg_{key}': 0 for key in self.stats_keys})

        self._metadata = {}
        self._metadata.update({f'total_{key}': 0 for key in self.stats_keys})
        self._metadata.update({'count_page': 0,
                               'count_paragraph': 0,
                               'count_line': 0,
                               'count_word': 0,
                               'count_char': 0})

    def _update_text_stats(self, text):
        """
        Updates text statistics based on the input text.

        Parameters
        ----------
        text : str
            Text for updating statistical metadata.
        """

        def update_stats(key, value):
            self.metadata[f'min_{key}'] = min(self.metadata[f'min_{key}'], value)
            self.metadata[f'max_{key}'] = max(self.metadata[f'max_{key}'], value)
            self._metadata[f'total_{key}'] += value

        paragraphs_in_page = text.split('\n\n')
        lines_in_page = text.split('\n')
        words_in_page = text.split()
        chars_in_page = text.replace('\n', '')

        for paragraph in paragraphs_in_page:
            lines_in_paragraph = paragraph.split('\n')
            words_in_paragraph = paragraph.split()
            chars_in_paragraph = paragraph.replace('\n', '')

            update_stats('lines_per_paragraph', len(lines_in_paragraph))
            update_stats('words_per_paragraph', len(words_in_paragraph))
            update_stats('chars_per_paragraph', len(chars_in_paragraph))

            for line in lines_in_paragraph:
                words_in_line = line.split()

                update_stats('words_per_line', len(words_in_line))
                update_stats('chars_per_line', len(line))

                for word in words_in_line:
                    update_stats('chars_per_word', len(word))

        update_stats('paragraphs_per_page', len(paragraphs_in_page))
        update_stats('lines_per_page', len(lines_in_page))
        update_stats('words_per_page', len(words_in_page))
        update_stats('chars_per_page', len(chars_in_page))

        self._metadata['count_page'] += 1
        self._metadata['count_paragraph'] += len(paragraphs_in_page)
        self._metadata['count_line'] += len(lines_in_page)
        self._metadata['count_word'] += len(words_in_page)
        self._metadata['count_char'] += len(chars_in_page)

        for key in self.stats_keys:
            count_key = 'count_' + key.split('_')[-1]

            if self._metadata[count_key] > 0:
                self.metadata[f'avg_{key}'] = round(
                    self._metadata[f'total_{key}'] / self._metadata[count_key])

        self.lexical_shape = (
            self.metadata['max_paragraphs_per_page'],
            self.metadata['max_lines_per_paragraph'],
            self.metadata['max_chars_per_line'],
            len(self.chars) + 1,
        )

    def encode_text(self, text, keepstats=False):
        """
        Encode text into a nested list of character indices.

        Parameters
        ----------
        text : str
            The text to be encoded.
        keepstats : bool, optional
            If True, updates the character set.

        Returns
        -------
        list of list of list of int
            The encoded representation of the text.
        """

        if keepstats:
            for word in set(text.replace('\n', ' ').split()):
                if word not in self.words:
                    self.words.append(word)

            for char in set(text.replace('\n', '')):
                if char not in self.chars:
                    self.chars.append(char)

            self._update_text_stats(text)

        char_to_index = {char: idx for idx, char in enumerate(self.chars)}

        pad_idx = char_to_index.get(self.pad_tk)
        sos_idx = char_to_index.get(self.sos_tk)
        eos_idx = char_to_index.get(self.eos_tk)
        unk_idx = char_to_index.get(self.unk_tk)

        text = [x.split('\n') for x in text.split('\n\n')]

        max_length = max(len(line) for paragraph in text for line in paragraph)
        padding_length = max_length + 2

        encoded_text = []
        for paragraph in text:

            encoded_paragraph = []
            for line in paragraph:
                encoded_line = [char_to_index.get(char, unk_idx) for char in line]
                encoded_line = [sos_idx] + encoded_line + [eos_idx]

                padding = (padding_length - len(encoded_line))
                encoded_paragraph.append(encoded_line + [pad_idx] * padding)

            encoded_text.append(encoded_paragraph)

        return encoded_text

    def decode_text(self, encoded_text):
        """
        Decode the encoded text back to its original form.

        Parameters
        ----------
        encoded_text : list of list of list of int
            The encoded text to be decoded.

        Returns
        -------
        str
            The decoded text.
        """

        translation_table = str.maketrans('', '', ''.join(self.chars[:4]))
        index_to_char = {idx: char for idx, char in enumerate(self.chars)}

        decoded_text = []
        for encoded_paragraph in encoded_text:

            decoded_paragraph = []
            for encoded_line in encoded_paragraph:
                decoded_line = [index_to_char.get(encoded_char, '') for encoded_char in encoded_line]
                decoded_paragraph.append(''.join(decoded_line))

            decoded_text.append('\n'.join(decoded_paragraph))

        decoded_text = '\n\n'.join(decoded_text)
        decoded_text = decoded_text.translate(translation_table)

        return decoded_text

    def encode_writer(self, writer, keepstats=False):
        """
        Encode a writer's ID into an integer index.

        Parameters
        ----------
        writer : str
            The writer's ID to encode.
        keepstats : bool, optional
            If True, updates the writer set.

        Returns
        -------
        int
            The encoded index of the writer.
        """

        if keepstats and writer not in self.writers:
            self.writers.append(writer)

        writer_to_index = {writer: idx for idx, writer in enumerate(self.writers)}
        encoded_writer = writer_to_index.get(writer, writer_to_index.get(self.unk_tk))

        return encoded_writer

    def decode_writer(self, encoded_writer):
        """
        Decode the encoded writer index back to the writer's ID.

        Parameters
        ----------
        encoded_writer : int
            The encoded index of the writer.

        Returns
        -------
        str
            The decoded writer's ID.
        """

        index_to_writer = {idx: char for idx, char in enumerate(self.writers)}
        writer = index_to_writer.get(encoded_writer, self.unk_tk)

        return writer
