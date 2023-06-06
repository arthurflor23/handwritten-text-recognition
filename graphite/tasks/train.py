from dataset import Augmentor, Dataset
from model import LanguageModel, OpticalModel


def train(args):

    dataset = Dataset(source=args.source,
                      level=args.level,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
                      seed=42)

    augmentor = Augmentor(erosion=args.erosion,
                          dilation=args.dilation,
                          elastic_transform=args.elastic_transform,
                          mixup=args.mixup,
                          perspective_transform=args.perspective_transform,
                          salt_and_pepper=args.salt_and_pepper,
                          gaussian_blur=args.gaussian_blur,
                          shearing=args.shearing,
                          scaling=args.scaling,
                          rotation=args.rotation,
                          translation=args.translation,
                          reference_pixels=dataset.reference_pixels,
                          seed=42)

    language_model = LanguageModel(env_key='OPENAI_API_KEY')

    optical_model = OpticalModel(network=args.network, seed=42)

    # carbon = Carbon(dataset=dataset,
    #                 augmentor=augmentor,
    #                 spell_checker=spell_checker,
    #                 optical_model=optical_model)
