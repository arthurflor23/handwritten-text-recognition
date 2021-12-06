"""
Language Model class.
Create and read the corpus with the language model file.
"""

import os
import re
import string

from kaldiio import WriteHelper


class LanguageModel():

    def __init__(self, output, N=3):
        self.output_path = os.path.join(output, "language")
        self.N = N

    def generate_kaldi_assets(self, dtgen, predicts):
        # get data and ground truth lists
        ctc_TK, space_TK, ground_truth = "<ctc>", "<space>", []

        for pt in ['train', 'valid', 'test']:
            for x in dtgen.dataset[pt]['gt']:
                ground_truth.append([space_TK if y == " " else y for y in list(f" {x} ")])

        # define dataset size and default tokens
        ds_size = dtgen.size['train'] + dtgen.size['valid'] + dtgen.size['test']

        # get chars list and save with the ctc and space tokens
        chars = list(dtgen.tokenizer.chars) + [ctc_TK]
        chars[chars.index(" ")] = space_TK

        kaldi_path = os.path.join(self.output_path, "kaldi")
        os.makedirs(kaldi_path, exist_ok=True)

        with open(os.path.join(kaldi_path, "chars.lst"), "w") as lg:
            lg.write("\n".join(chars))

        ark_file_name = os.path.join(kaldi_path, "conf_mats.ark")
        scp_file_name = os.path.join(kaldi_path, "conf_mats.scp")

        # save ark and scp file (laia output/kaldi input format)
        with WriteHelper(f"ark,scp:{ark_file_name},{scp_file_name}") as writer:
            for i, item in enumerate(predicts):
                writer(str(i + ds_size), item)

        # save ground_truth.lst file with sparse sentences
        with open(os.path.join(kaldi_path, "ground_truth.lst"), "w") as lg:
            for i, item in enumerate(ground_truth):
                lg.write(f"{i} {' '.join(item)}\n")

        # save indexes of the train/valid and test partitions
        with open(os.path.join(kaldi_path, "ID_train.lst"), "w") as lg:
            range_index = [str(i) for i in range(0, ds_size - dtgen.size['test'])]
            lg.write("\n".join(range_index))

        with open(os.path.join(kaldi_path, "ID_test.lst"), "w") as lg:
            range_index = [str(i) for i in range(ds_size - dtgen.size['test'], ds_size)]
            lg.write("\n".join(range_index))

    def kaldi(self, predict=True):
        """
        Kaldi Speech Recognition Toolkit with SRI Language Modeling Toolkit.
        ** Important Note **
        You'll need to do all by yourself:
        1. Compile Kaldi with SRILM and OpenBLAS.
        2. Create and add kaldi folder in the project `lib` folder (``src/lib/kaldi/``)
        3. Generate files (search `--kaldi_assets` in https://github.com/arthurflor23/handwritten-text-recognition):
            a. `chars.lst`
            b. `conf_mats.ark`
            c. `ground_truth.lst`
            d. `ID_test.lst`
            e. `ID_train.lst`
        4. Add files (item 3) in the project `output` folder: ``output/<DATASET>/kaldi/``
        More information (maybe help) in ``src/lib/kaldi-decode-script.sh`` comments.
        References:
            D. Povey, A. Ghoshal, G. Boulianne, L. Burget, O. Glembek, N. Goel, M. Hannemann,
            P. Motlicek, Y. Qian, P. Schwarz, J. Silovsky, G. Stem- mer and K. Vesely.
            The Kaldi speech recognition toolkit, 2011.
            Workshop on Automatic Speech Recognition and Understanding.
            URL: http://github.com/kaldi-asr/kaldi
            Andreas Stolcke.
            SRILM - An Extensible Language Modeling Toolkit, 2002.
            Proceedings of the 7th International Conference on Spoken Language Processing (ICSLP).
            URL: http://www.speech.sri.com/projects/srilm/
        """

        option = "TEST" if predict else "TRAIN"
        output = os.path.join(self.output_path, "kaldi")

        if os.system(f"./language/kaldi-decode-script.sh {output} {option} {self.N}") != 0:
            print("\n##################\n")
            print("Kaldi script error.")
            print("\n##################\n")

        if predict:
            predicts = open(os.path.join(output, "data", "predicts_t")).read().splitlines()

            for i, line in enumerate(predicts):
                tokens = line.split()
                predicts[i] = "".join(tokens[1:]).replace("<space>", " ").strip()

            return predicts
