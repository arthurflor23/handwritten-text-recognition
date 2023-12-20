import os
import cv2
import json

from data import Dataset
from models import Graphite


def inference(args):
    """
    Executes the inference phase.

    Parameters
    ----------
    args : argparse.Namespace
        A namespace object containing all the arguments required.
    """

    tokenizer, context = Graphite().get_tokenizer(synthesis=args.synthesis,
                                                  synthesis_index=args.synthesis_index,
                                                  recognition=args.recognition,
                                                  recognition_index=args.recognition_index,
                                                  experiment_name=args.experiment_name)

    if tokenizer is None or context is None:
        print('Tokenizer or context not found to load.')
        return

    data = {
        'test': [{
            'image': args.image,
            'bbox': args.bbox,
            'text': args.text,
            'writer': '1',
        }],
    }

    dataset = Dataset(data=data,
                      image_shape=args.image_shape,
                      lazy_mode=args.lazy_mode,
                      tokenizer=tokenizer,
                      seed=args.seed)
    print(dataset)

    graphite = Graphite(workflow=args.workflow,
                        synthesis=args.synthesis,
                        recognition=args.recognition,
                        spelling=args.spelling,
                        image_shape=args.image_shape,
                        tokenizer=dataset.tokenizer,
                        synthesis_ratio=args.synthesis_ratio,
                        experiment_name=args.experiment_name)
    print(graphite)

    graphite.compile(learning_rate=args.learning_rate, context=context)

    basename = os.path.splitext(os.path.basename(args.image or ''))[0]
    os.makedirs(args.output, exist_ok=True)

    if 'recognition' in args.workflow:
        prediction_configs = [{
            'predict': True,
            'corrections': False,
        }, {
            'predict': args.spelling,
            'corrections': True,
        }]

        for config in prediction_configs:
            if not config['predict']:
                continue

            infer_gen, infer_steps = dataset.get_generator(data_partition='test',
                                                           batch_size=args.batch_size)

            predictions, probabilities = graphite.predict_recognition(x=infer_gen,
                                                                      steps=infer_steps,
                                                                      top_paths=args.top_paths,
                                                                      beam_width=args.beam_width,
                                                                      ctc_decode=True,
                                                                      token_decode=True,
                                                                      corrections=config['corrections'])

            content = []
            for x, p in zip(predictions, probabilities):
                for i in range(len(x)):
                    content.append({
                        'top_path': i+1,
                        'probability': p[i],
                        'prediction': x[i],
                    })

            content = json.dumps(content, indent=4, sort_keys=False)
            print(content)

            filepath = os.path.join(args.output, f"{basename}.json")
            with open(filepath, 'w') as f:
                f.write(content)

    elif 'synthesis' in args.workflow:
        infer_gen, infer_steps = dataset.get_generator(data_partition='test',
                                                       batch_size=args.batch_size)

        predictions = graphite.predict_synthesis(x=infer_gen, steps=infer_steps)

        style = 'guided_style' if args.image else 'random_style'
        filepath = f"{basename}_{style}".strip('_')

        for i, image in enumerate(predictions):
            generated_filepath = os.path.join(args.output, f"{filepath}.png")
            cv2.imwrite(generated_filepath, image)
