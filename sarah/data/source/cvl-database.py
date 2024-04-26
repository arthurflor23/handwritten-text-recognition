import os
import re
import glob
import string
import xml.etree.ElementTree as ET


class Source():
    """
    Represents the CVL-Database database source.

    Requires implementation of `fetch_data`, returning a dictionary with
        'training', 'validation', and 'test' keys, each mapping to a list of data entries.

    Each data entry is a dictionary with keys 'image', 'bbox', 'text', and 'writer':
        'image' : str
            path to the image.
        'bbox' : list
            bounding box coordinates [x, y, h, w] (empty if no bbox).
        'text' : str
            text content with '\n' as line breaks.
        'writer' : str
            writer's unique ID ('1' for unique writer).
    """

    def __init__(self, artifact_path):
        """
        Initializes a new instance of the Source class.

        Parameters
        ----------
        artifact_path : str
            The path to the data.
        """

        self.artifact_path = artifact_path
        self.base_path = os.path.join(self.artifact_path, 'cvl-database')

        self.training_path = os.path.join(self.base_path, 'trainset')
        self.test_path = os.path.join(self.base_path, 'testset')

        self.ns = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2010-03-19'}

    def fetch_data(self, text_level):
        """
        Retrieves the data for training, validation, and test partitions.

        Parameters
        ----------
        text_level : str
            The granularity of the data to be retrieved.

        Returns
        -------
        dict
            Partition dictionary with list of items.
        """

        data = {'training': [], 'validation': [], 'test': []}

        if not os.path.isdir(self.base_path):
            return data

        if text_level == 'word':
            data['training'] = self._load_data(self.training_path, self._extract_words)
            data['test'] = self._load_data(self.test_path, self._extract_words)

        elif text_level == 'line':
            data['training'] = self._load_data(self.training_path, self._extract_lines)
            data['test'] = self._load_data(self.test_path, self._extract_lines)

        elif text_level == 'paragraph':
            data['training'] = self._load_data(self.training_path, self._extract_paragraphs)
            data['test'] = self._load_data(self.test_path, self._extract_paragraphs)

        return data

    def _load_data(self, partition_path, extract_function):
        """
        Load data from a partition.

        Parameters
        ----------
        partition_path : str
            Path to the partition.
        extract_function : function
            Function to extract data.

        Returns
        -------
        list
            List of extracted data.
        """

        data = []
        exclude_data = ['0992-1.tif']

        xml_files = glob.glob(os.path.join(partition_path, 'xml', '*.xml'))

        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            filename = root.find('.//ns:Page', self.ns).get('imageFilename')

            if filename and filename not in exclude_data:
                data.extend(extract_function(root, partition_path, filename))

        return data

    def _extract_words(self, root, partition_path, filename):
        """
        Loads the words data.

        Parameters
        ----------
        root : ElementTree.Element
            Root element of XML.
        partition_path : str
            Path to the partition.
        filename : str
            Filename of the XML.

        Returns
        -------
        list
            List of word data.
        """

        writer_id = filename.split('-')[0]
        words_data = []

        for attr_region in root.findall('.//ns:AttrRegion[@attrType="1"]', self.ns):
            text = attr_region.get('text', '').strip(f" {string.punctuation}")

            if text:
                attr_id = attr_region.get('id')
                images_pattern = os.path.join(partition_path, 'words', writer_id, f'{attr_id}*')
                images = glob.glob(images_pattern)

                if images:
                    words_data.append({
                        'image': images[0],
                        'bbox': [],
                        'text': text,
                        'writer': writer_id,
                    })

        return words_data

    def _extract_lines(self, root, partition_path, filename):
        """
        Loads the lines data.

        Parameters
        ----------
        root : ElementTree.Element
            Root element of XML.
        partition_path : str
            Path to the partition.
        filename : str
            Filename of the XML.

        Returns
        -------
        list
            List of line data.
        """

        writer_id = filename.split('-')[0]
        lines_data = []

        for attr_region in root.findall('.//ns:AttrRegion[@attrType="3"]', self.ns):
            for line_region in attr_region.findall('.//ns:AttrRegion[@attrType="2"]', self.ns):
                text = ' '.join(
                    word.get('text') for word in
                    line_region.findall('.//ns:AttrRegion[@attrType="1"]', self.ns) if word.get('text')
                ).strip()

                if text:
                    text = self._apply_punctuation_rules(filename, text)

                    attr_id = line_region.get('id')
                    image_path = os.path.join(partition_path, 'lines', writer_id, f'{attr_id}.tif')

                    lines_data.append({
                        'image': image_path,
                        'bbox': [],
                        'text': text,
                        'writer': writer_id,
                    })

        return lines_data

    def _extract_paragraphs(self, root, partition_path, filename):
        """
        Loads the paragraphs data.

        Parameters
        ----------
        root : ElementTree.Element
            Root element of XML.
        partition_path : str
            Path to the partition.
        filename : str
            Filename of the XML.

        Returns
        -------
        list
            List of paragraph data.
        """

        writer_id = filename.split('-')[0]
        paragraphs_data = []

        for attr_region in root.findall('.//ns:AttrRegion[@attrType="3"]', self.ns):
            text = '\n'.join(
                ' '.join(word.get('text') for word in line_region.findall(
                    './/ns:AttrRegion[@attrType="1"]', self.ns) if word.get('text'))
                for line_region in attr_region.findall('.//ns:AttrRegion[@attrType="2"]', self.ns)
            ).replace('\n\n', '\n').strip()

            if text:
                text = self._apply_punctuation_rules(filename, text)
                image_path = os.path.join(partition_path, 'paragraphs', filename)

                paragraphs_data.append({
                    'image': image_path,
                    'bbox': [],
                    'text': text,
                    'writer': writer_id,
                })

        return paragraphs_data

    def _apply_punctuation_rules(self, filename, text):
        """
        Applies punctuation rules to the text.

        Parameters
        ----------
        filename : str
            Filename of the XML.
        text : str
            Text to apply punctuation rules to.

        Returns
        -------
        str
            Text with applied punctuation rules.
        """

        rules = {
            '1': [
                {'Lines': 'Lines,'},
                {'Triangles': 'Triangles,', 'angles Squares': 'angles, Squares'},
                {'Squares': 'Squares,'},
                {'Pentagons': 'Pentagons,', 'gons Hexagons': 'gons, Hexagons'},
                {'Hexagons': 'Hexagons,', 'gons and': 'gons, and'},
                {'figures': 'figures,', 'ures instead': 'ures, instead'},
                {'places': 'places,'},
                {'about': 'about,'},
                {'surface': 'surface,', 'face but': 'face, but', 'ace but': 'ace, but'},
                {'it': 'it,'},
                {'shadows': 'shadows -', 'dows only': 'dows - only'},
                {'edges': 'edges -'},
                {'countrymen': 'countrymen.', 'men Alas': 'men. Alas'},
                {'Alas': 'Alas,'},
                {'ago': 'ago,'},
                {'said my': 'said "my', 'my universe': '"my universe'},
                {'universe': 'universe":', 'verse but': 'verse" but'},
                {'things': 'things.'},
            ],
            '2': [
                {'fortune on': 'fortune, on'},
                {'smiling': 'smiling.'},
                {'whore': 'whore:'},
                {'weak': 'weak:'},
                {'Nomacs': 'Nomacs -'},
                {'fortune with': 'fortune, with', 'Disdaining fortune': 'Disdaining fortune,'},
                {'steel': 'steel.'},
                {'execution': 'execution.', 'cution': 'cution.'},
                {'slave': 'slave;'},
            ],
            '3': [
                {'Mailfterl is': 'Mailüterl is'},
                {'mainland': 'mainland.', 'land': 'land.'},
                {'Zemanek': 'Zemanek.'},
                {'computer': 'computer:'},
                {'If': '"If'},
                {'achieve': 'achievem,', 'ieve': 'ieve,'},
                {'Mailfterl': 'Mailüterl".', 'lfterl The': 'lüterl". The'},
                {'Rechenautomat': 'Rechenautomat.', 'automat': 'automat.', 'tomat': 'tomat.', 'mat': 'mat.'},
                {' -Rechenautomat': 'Rechenautomat', '-Rechenautomat': 'Rechenautomat'},
                {'Volltransistor- ': 'Volltransistor-', 'Volltransistor--': 'Volltransistor-'},
            ],
            '4': [
                {'animals': 'animals,', 'mals': 'mals,'},
                {'us': 'us,'},
                {'is': 'is,'},
                {'other': 'other,'},
                {'nature': 'nature.'},
            ],
            '6': [
                {'Werd ich': 'Werd\' ich'},
                {'sagen': 'sagen:'},
                {'doch': 'doch!'},
                {'schn': 'schön!'},
                {'schlagen': 'schlagen,'},
                {'gehn': 'gehn!'},
                {'schallen': 'schallen,'},
                {'frey': 'frey,'},
                {'stehn': 'stehn,'},
                {'fallen': 'fallen,'},
                {'Zeit fr': 'Zeit für', 'fr mich': 'für mich'},
                {'vorbey': 'vorbey!'},
            ],
            '7': [
                {'love': 'love.'},
                {'imagination': 'imagination.', 'ation': 'ation.'},
                {'curiosity': 'curiosity.'},
                {'effect': 'effect.'},
                {'marvellous': 'marvellous,'},
                {'intellect': 'intellect,'},
                {'art': 'art.'},
                {'away': 'away.'},
                {'stupid': 'stupid.'},
            ],
            '8': [
                {'gazed': 'gazed,'},
                {'widened': 'widened -'},
                {'whirlwind': 'whirlwind -', 'wind the': 'wind - the'},
                {'sight': 'sight -'},
                {'asunder': 'asunder -'},
                {'waters': 'waters -'},
                {'House': '"House'},
                {'Usher': 'Usher".'},
            ],
        }

        def replace_term(text, term, replacement):
            pattern = r'(\W|^)' + re.escape(term) + r'(\W|$)'
            regex = re.compile(pattern, re.IGNORECASE)

            def replace_func(match):
                return match.group(1) + replacement + match.group(2)

            return regex.subn(replace_func, text, 1)

        rule = os.path.splitext(filename)[0].split('-')[-1]

        for group in rules.get(rule, []):
            for term, replacement in group.items():
                text, count = replace_term(text, term, replacement)

                if count > 0:
                    break

        return text
