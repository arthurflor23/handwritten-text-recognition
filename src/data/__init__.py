from importlib import import_module


class Data():

    def __init__(self, args, jsonf):
        dirs = jsonf[args.module][args.dataset]
        self.input_dir = dirs["input_dir"]
        self.input_gt_dir = dirs["input_gt_dir"]
        self.partition_dir = dirs["partition_dir"]
        self.data_proc_dir = dirs["data_proc_dir"]
        self.model_dir = dirs["model_dir"]

        self.module = args.module
        self.dataset = args.dataset
        self.train = None
        self.train_gt = None
        self.validation = None
        self.validation_gt = None
        self.test = None
        self.test_gt = None

        mod = import_module(f"{__name__}.{self.dataset}")

        if(self.module == "dls"):
            self.load_train = mod.load_train_pages
            self.load_validation = mod.load_validation_pages
            self.load_test = mod.load_test_pages
        elif(self.module == "htr"):
            self.load_train = mod.load_train_lines
            self.load_validation = mod.load_validation_lines
            self.load_test = mod.load_test_lines
        elif(self.module == "nlp"):
            self.load_train = mod.load_train_txt
            self.load_validation = mod.load_validation_txt
            self.load_test = mod.load_test_txt

    def preprocess(self):
        mod = import_module(f"{self.module}.preproc")

        self.train, self.train_gt = mod.preprocess(self.train, self.train_gt)
        self.validation, self.validation_gt = mod.preprocess(
            self.validation, self.validation_gt)
        self.test, self.test_gt = mod.preprocess(self.test, self.test_gt)

    def imread_train(self):
        self.train, self.train_gt = self.load_train(self)
        self.validation, self.validation_gt = self.load_validation(self)

    def imread_test(self):
        self.test, self.test_gt = self.load_test(self)

    def imread_dataset(self):
        self.imread_train()
        self.imread_test()

    def imwrite(self):
        print("imwrite")
