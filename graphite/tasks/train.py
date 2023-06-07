from carbon import Carbon
from dataset import Augmentor, Dataset
from model import Model
from spelling import Spelling


def train(args):

    dataset = Dataset(source=args.source,
                      level=args.level,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
                      seed=42)

    # augmentor = Augmentor(erosion=args.erosion,
    #                       dilation=args.dilation,
    #                       elastic_transform=args.elastic_transform,
    #                       mixup=args.mixup,
    #                       perspective_transform=args.perspective_transform,
    #                       salt_and_pepper=args.salt_and_pepper,
    #                       gaussian_blur=args.gaussian_blur,
    #                       shearing=args.shearing,
    #                       scaling=args.scaling,
    #                       rotation=args.rotation,
    #                       translation=args.translation,
    #                       reference_pixels=dataset.reference_pixels,
    #                       seed=42)

    # spelling = Spelling(spell_checker=args.spell_checker,
    #                     api_key=args.api_key,
    #                     env_key=args.env_key)

    print(dataset.tokenizer.shape)

    model = Model(network=args.network,
                  network_flavor=args.network_flavor,
                  seed=42)

    # carbon = Carbon(dataset=dataset,
    #                 augmentor=augmentor,
    #                 spelling=spelling,
    #                 optical_model=optical_model)
