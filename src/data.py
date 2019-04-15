from glob import glob
import os


class Data():

    def __init__(self, args):
        self.input_dir = args.input_dir
        self.gt_dir = args.gt_dir

        self.subsets_dir = args.subsets_dir
        self.output_dir = args.output_dir

        self.train, self.train_gt = None, None
        self.valid, self.valid_gt = None, None
        self.test, self.test_gt = None, None

    def load_dataset(self):
        self.train, self.train_gt = self.load_subset("train")
        # self.valid, self.valid_gt = self.load_subset("valid")
        # self.test, self.test_gt = self.load_subset("test")

    def load_subset(self, subset):
        subset_f = os.path.join(self.subsets_dir, f"{subset}.txt")

        with open(subset_f) as f:
            content = f.readlines()
            content = [x.strip() for x in content]
            f.close()

        gt_files = os.listdir(self.gt_dir)
        print(len(gt_files))

        if (len(gt_files) > 0):
            

        # with open(subset_f) as f:
        #     content = f.readlines()
        #     content = [x.strip() for x in content]
        #     f.close()

        for item in content:
            s = item.split("-")
            dt_path = os.path.join(
                self.input_dir, s[0], f"{s[0]}-{s[1]}", f"{item}.png")

            # print(dt_path)

        return [], []
