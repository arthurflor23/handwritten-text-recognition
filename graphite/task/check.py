from dataset import Dataset


def check(args):

    # print("check data", args)

    dataset = Dataset(source=args.source,
                      level=args.level,
                      training_ratio=args.training_ratio,
                      validation_ratio=args.validation_ratio,
                      test_ratio=args.test_ratio,
                      lazy_mode=args.lazy_mode,
                      data_path=args.data_path,
                      seed=42)

    # print(dataset)
