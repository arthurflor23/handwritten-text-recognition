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
                          disable_augmentation=args.disable_augmentation,
                          seed=42)

    spelling = Spelling(spell_checker=args.spell_checker, api_key=args.api_key, env_key=args.env_key)

    model = Model(network=args.network, tokenizer=dataset.tokenizer, seed=42)
    model.compile(learning_rate=args.learning_rate, model_uri=None)

    # model.fit(...,
    #           plateau_cooldown=args.plateau_cooldown,
    #           plateau_factor=args.plateau_factor,
    #           plateau_patience=args.plateau_patience,
    #           patience=args.patience)

    # carbon = Carbon(dataset=dataset,
    #                 augmentor=augmentor,
    #                 spelling=spelling,
    #                 optical_model=optical_model)
