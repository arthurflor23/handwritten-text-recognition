import argparse
import cv2
import os
import string

from data import preproc as pp
from data.generator import Tokenizer

from network.model import HTRModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--arch", type=str, default="flor")

    parser.add_argument("--archive", type=bool, default=False)
    parser.add_argument("--csv", type=str, default="")
    parser.add_argument("--parquet", type=str, default="")

    args = parser.parse_args()

    source_path = args.source
    weights_path = args.weights

    input_size = (1024, 128, 1)
    max_text_length = 128
    charset_base = string.printable[:95]

    if args.image:
        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)

        img = pp.preprocess(args.source, input_size=input_size)
        x_test = pp.normalization([img])

        model = HTRModel(architecture=args.arch,
                            input_size=input_size,
                            vocab_size=tokenizer.vocab_size,
                            beam_width=10,
                            top_paths=10)

        model.compile(learning_rate=0.001)
        model.load_checkpoint(target=weights_path)

        predicts, probabilities = model.predict(x_test, ctc_decode=True)
        predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

        print("\n####################################")
        for i, (pred, prob) in enumerate(zip(predicts, probabilities)):
            print("\nProb.  - Predict")

            for (pd, pb) in zip(pred, prob):
                print(f"{pb:.4f} - {pd}")

            cv2.imshow(f"Image {i + 1}", cv2.imread(args.image))
        print("\n####################################")
        cv2.waitKey(0)
    elif args.directory:
        tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)
        model = HTRModel(architecture=args.arch,
                            input_size=input_size,
                            vocab_size=tokenizer.vocab_size,
                            beam_width=10,
                            top_paths=10)

        model.compile(learning_rate=0.001)
        model.load_checkpoint(target=weights_path)

        images = [x for x in os.listdir(args.source) if x.split(".")[-1] == "jpg" or x.split(".")[-1] == "jp2"]
        for img in images:
            img = pp.preprocess(args.image, input_size=input_size)
            x_test = pp.normalization([img])

            predicts, probabilities = model.predict(x_test, ctc_decode=True)
            predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

            for i, (pred, prob) in enumerate(zip(predicts, probabilities)):

                for (pd, pb) in zip(pred, prob):
                    print(f"{pb:.4f} - {pd}")

                cv2.imshow(f"Image {i + 1}", cv2.imread(args.image))
            cv2.waitKey(0)
    elif args.archive:
        # tokenizer = Tokenizer(chars=charset_base, max_text_length=max_text_length)

        # img = pp.preprocess(args.image, input_size=input_size)
        # x_test = pp.normalization([img])

        # model = HTRModel(architecture=args.arch,
        #                     input_size=input_size,
        #                     vocab_size=tokenizer.vocab_size,
        #                     beam_width=10,
        #                     top_paths=10)

        # model.compile(learning_rate=0.001)
        # model.load_checkpoint(target=target_path)

        # predicts, probabilities = model.predict(x_test, ctc_decode=True)
        # predicts = [[tokenizer.decode(x) for x in y] for y in predicts]

        # print("\n####################################")
        # for i, (pred, prob) in enumerate(zip(predicts, probabilities)):
        #     print("\nProb.  - Predict")

        #     for (pd, pb) in zip(pred, prob):
        #         print(f"{pb:.4f} - {pd}")

        #     cv2.imshow(f"Image {i + 1}", cv2.imread(args.image))
        # print("\n####################################")
        # cv2.waitKey(0)
        pass