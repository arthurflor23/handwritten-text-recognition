import re


class Tokenizer():
    """
    Tokenizer for text, writer IDs, and graph shape metadata.
    """

    def __init__(self):
        """
        Initializes the tokenizer.
        """

        self.pad_tk = '¶'
        self.sos_tk = '◖'
        self.eos_tk = '◗'

        self.chars = [self.pad_tk, self.sos_tk, self.eos_tk]
        self.words = []
        self.writers = []

        self.lexical_shape = []
        self.writers_shape = []

        self.special_chars_length = len(self.chars)
        self.special_marks_length = self.special_chars_length - 1

        self._initialize_metadata()

    def __repr__(self):
        """
        Returns a formatted tokenizer summary.

        Returns
        -------
        str
            Formatted summary.
        """

        pad, width = 25, 68
        chunk_length = width - pad - 4

        info = "=" * width
        info += f'\n{self.__class__.__name__.center(width)}'
        info += "\n" + "-" * width
        info += f"\n{'words':<{pad}}: {len(self.words):,}"
        info += f"\n{'chars':<{pad}}: {len(self.chars) - self.special_chars_length:,}"
        info += f"\n{'writers':<{pad}}: {len(self.writers):,}"
        info += "\n" + "-" * width
        info += f"\n{'writers_shape':<{pad}}: {self.writers_shape}"
        info += f"\n{'lexical_shape':<{pad}}: {self.lexical_shape}"
        info += f"\n{'font_graph_shape':<{pad}}: {self.font_graph_shape or '-'}"
        info += "\n" + "-" * width

        chars = ''.join(self.chars)
        chunks = [chars[i:i+chunk_length] for i in range(0, len(chars), chunk_length)]

        info += f"\n{'charset':<{pad}}: {chunks[0]}"
        for chunk in chunks[1:]:
            info += f"\n{'':<{pad}}  {chunk}"

        info += "\n" + "-" * width

        for key, value in self.metadata.items():
            info += f"\n{key:<{pad}}: {value:,}"

        return info

    def _initialize_metadata(self):
        """
        Initializes text statistics and shape metadata.
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

        self.font_graph_shape = None

    def _update_text_stats(self, text):
        """
        Updates text statistics.

        Parameters
        ----------
        text : str
            Input text.
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

    def encode_text(self, text, keepstats=False):
        """
        Encodes text into character indices.

        Parameters
        ----------
        text : str
            Input text.
        keepstats : bool, optional
            Whether to update text statistics and vocabulary.

        Returns
        -------
        list
            Encoded text.
        """

        if keepstats:
            self._update_text_stats(text)

            words = sorted(set(text.replace('\n', ' ').split()), reverse=False)
            chars = sorted(set(text.replace('\n', ' ')), reverse=False)

            for word in words:
                if word not in self.words:
                    self.words.append(word)

            for char in chars:
                if char not in self.chars:
                    self.chars.append(char)

            def next_power_of_two(n):
                return 1 if n == 0 else 2 ** (n - 1).bit_length()

            self.font_graph_shape = None

            self.lexical_shape = (
                next_power_of_two(self.metadata['max_lines_per_page']),
                next_power_of_two(self.metadata['max_chars_per_line'] + self.special_marks_length),
                len(self.chars) + 1,
            )

        char_to_index = {char: idx for idx, char in enumerate(self.chars)}

        pad_idx = char_to_index.get(self.pad_tk)
        sos_idx = char_to_index.get(self.sos_tk)
        eos_idx = char_to_index.get(self.eos_tk)

        text = [' '.join(x.split()) for x in re.sub(r'\n\n+', '\n', text).split('\n')]
        max_length = max([len(line) for line in text]) + self.special_marks_length

        encoded_text = []
        for line in text:
            encoded_line = [char_to_index.get(char, 0) for char in line]
            encoded_line = [sos_idx] + encoded_line + [eos_idx]

            padding = (max_length - len(encoded_line))
            encoded_text.append(encoded_line + ([pad_idx] * padding))

        return encoded_text

    def decode_text(self, encoded_text):
        """
        Decodes character indices into text.

        Parameters
        ----------
        encoded_text : list
            Encoded text.

        Returns
        -------
        str
            Decoded text.
        """

        translation_table = str.maketrans('', '', ''.join(self.chars[:self.special_chars_length]))
        index_to_char = {idx: char for idx, char in enumerate(self.chars)}

        decoded_text = []
        for encoded_line in encoded_text:
            decoded_line = [index_to_char.get(encoded_char, '') for encoded_char in encoded_line]
            decoded_text.append(''.join(decoded_line))

        decoded_text = '\n'.join(decoded_text)
        decoded_text = decoded_text.translate(translation_table)

        return decoded_text

    def encode_writer(self, writer, keepstats=False):
        """
        Encodes a writer ID into an index.

        Parameters
        ----------
        writer : str
            Writer ID.
        keepstats : bool, optional
            Whether to update the writer vocabulary.

        Returns
        -------
        int
            Encoded writer index.
        """

        if keepstats and writer not in self.writers:
            self.writers.append(writer)
            self.writers_shape = tuple([len(self.writers)])

        writer_to_index = {writer: idx for idx, writer in enumerate(self.writers)}
        encoded_writer = writer_to_index.get(writer, 0)

        return encoded_writer

    def decode_writer(self, encoded_writer):
        """
        Decodes a writer index into a writer ID.

        Parameters
        ----------
        encoded_writer : int
            Encoded writer index.

        Returns
        -------
        str
            Decoded writer ID.
        """

        index_to_writer = {idx: char for idx, char in enumerate(self.writers)}
        writer = index_to_writer.get(encoded_writer, 0)

        return writer

    def set_font_graph_shape(self, font_graphs):
        """
        Sets the maximum font graph shape.

        Parameters
        ----------
        font_graphs : list
            Font graph data.
        """

        if not font_graphs:
            return

        max_nodes = max(
            graph['nodes'].shape[0]
            for font in font_graphs
            for graph in font
            if len(graph['nodes']) > 0
        )

        max_edges = max(
            graph['edges'].shape[1]
            for font in font_graphs
            for graph in font
            if len(graph['edges']) > 0
        )

        nodes_shape = (max_nodes * len(self.chars), 4)
        edges_shape = (2, max_edges * len(self.chars))

        self.font_graph_shape = nodes_shape, edges_shape
