import os
import cv2
import json

from data import Dataset
from models import Compose


def inference(args):
    """
    Executes the inference phase.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object containing all the arguments required.
    """

    tokenizer, run_context = Compose().get_tokenizer(synthesis=args.synthesis,
                                                     synthesis_run_id=args.synthesis_run_id,
                                                     recognition=args.recognition,
                                                     recognition_run_id=args.recognition_run_id,
                                                     writer_identification=args.writer_identification,
                                                     writer_identification_run_id=args.writer_identification_run_id,
                                                     experiment_name=args.experiment_name,
                                                     finished_runs=args.finished_runs,
                                                     output_path=args.output_path)

    if tokenizer is None or run_context is None:
        print('Tokenizer or run context not found to load.')
        exit(1)

    data = {
        'test': [{
            'writer': '1',
            'image': args.image,
            'bbox': args.bbox,
            'text': args.text,
        }],
    }

    dataset = Dataset(data=data,
                      image_shape=args.image_shape,
                      pad_value=args.pad_value,
                      char_width=args.char_width,
                      mask_by_text=args.mask_by_text,
                      lazy_mode=args.lazy_mode,
                      tokenizer=tokenizer,
                      seed=args.seed)
    print(dataset)

    compose = Compose(synthesis=args.synthesis,
                      recognition=args.recognition,
                      spelling=args.spelling,
                      writer_identification=args.writer_identification,
                      image_shape=args.image_shape,
                      tokenizer=dataset.tokenizer,
                      experiment_name=args.experiment_name,
                      output_path=args.output_path,
                      gpu=args.gpu,
                      seed=args.seed)
    print(compose)

    compose.compile(learning_rate=args.learning_rate, run_context=run_context)

    infer_gen, infer_steps = dataset.get_generator(data_partition='test',
                                                   batch_size=args.batch_size)

    if args.writer_identification:
        predictions = compose.predict_writer_identification(x=infer_gen,
                                                            steps=infer_steps,
                                                            token_decode=True,
                                                            verbose=args.verbose)

        inferences = [
            {
                'index': i+1,
                'prediction': predictions[i],
            }
            for i in range(len(predictions))
        ]

        inferences = json.dumps(inferences, indent=4, ensure_ascii=False)
        print(inferences)

        basename = os.path.splitext(os.path.basename(args.image or ''))[0]
        filepath = os.path.join(args.output_path, f"{basename}.json")
        os.makedirs(args.output_path, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(inferences)

    elif args.recognition:
        predictions, probabilities = compose.predict_recognition(x=infer_gen,
                                                                 steps=infer_steps,
                                                                 top_paths=args.top_paths,
                                                                 beam_width=args.beam_width,
                                                                 ctc_decode=True,
                                                                 token_decode=True,
                                                                 verbose=args.verbose)

        if args.spelling:
            corrections = compose.predict_spelling(x=predictions, steps=infer_steps, verbose=args.verbose)

        inferences = [
            {
                'index': i+1,
                'top_path': y+1,
                'probability': probabilities[i][y],
                'prediction': predictions[i][y],
                'correction': corrections[i][y] if args.spelling else ''
            }
            for i in range(len(predictions))
            for y in range(len(predictions[i]))
        ]

        inferences = json.dumps(inferences, indent=4, ensure_ascii=False)
        print(inferences)

        basename = os.path.splitext(os.path.basename(args.image or ''))[0]
        filepath = os.path.join(args.output_path, f"{basename}.json")
        os.makedirs(args.output_path, exist_ok=True)

        with open(filepath, 'w') as f:
            f.write(inferences)

    elif args.synthesis:
        predictions = compose.predict_synthesis(x=infer_gen,
                                                steps=infer_steps,
                                                verbose=args.verbose)

        basename = os.path.splitext(os.path.basename(args.image or ''))[0]
        filepath = f"{basename}_{'guided' if args.image else 'random'}".strip('_')
        os.makedirs(args.output_path, exist_ok=True)

        for i, image in enumerate(predictions):
            generated_filepath = os.path.join(args.output_path, f"{filepath}_{i+1}.png")
            cv2.imwrite(generated_filepath, image)
