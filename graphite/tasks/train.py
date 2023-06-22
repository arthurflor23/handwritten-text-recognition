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
    print(dataset)

    augmentor = Augmentor(elastic_transform=args.elastic_transform,
                          erosion=args.erosion,
                          dilation=args.dilation,
                          mixup=args.mixup,
                          perspective_transform=args.perspective_transform,
                          salt_and_pepper=args.salt_and_pepper,
                          gaussian_blur=args.gaussian_blur,
                          shearing=args.shearing,
                          scaling=args.scaling,
                          rotation=args.rotation,
                          translation=args.translation,
                          disable_augmentation=args.disable_augmentation,
                          seed=42)
    print(augmentor)

    model = Model(network=args.network, tokenizer=dataset.tokenizer, seed=42)
    model.compile(learning_rate=args.learning_rate, run_index=args.run_index)

    train_data, train_steps = dataset.get_generator(dataset.training, batch_size=args.batch_size, augmentor=augmentor)
    valid_data, valid_steps = dataset.get_generator(dataset.validation, batch_size=args.batch_size, augmentor=None)

    model.fit(epochs=args.epochs,
              training_data=train_data,
              training_steps=train_steps,
              validation_data=valid_data,
              validation_steps=valid_steps,
              plateau_factor=args.plateau_factor,
              plateau_cooldown=args.plateau_cooldown,
              plateau_patience=args.plateau_patience,
              patience=args.patience,
              verbose=1)

    test_data, test_steps = dataset.get_generator(dataset.test, batch_size=16, augmentor=None)

    predicts, probabilities = model.predict(test_data=test_data,
                                            test_steps=test_steps,
                                            top_paths=args.top_paths,
                                            beam_width=args.beam_width,
                                            ctc_decode=True,
                                            token_decode=True,
                                            verbose=1)

    print(predicts)
    print(predicts.shape, probabilities.shape)

    # spelling = Spelling(spell_checker=args.spell_checker, api_key=args.api_key, env_key=args.env_key)
