# Handwritten Text Recognition: Graphite

This project aims to provide a comprehensive and end-to-end solution for Handwritten Text Recognition (HTR) using [Tensorflow](https://www.tensorflow.org/). It includes a tutorial and a set of tools for data processing, model training, testing, and inference. The HTR model can be trained on various datasets and supports different levels of recognition, such as line and paragraph level.

The project offers support for spell checkers through API integration, allowing for improved accuracy and error correction. It is also designed to be highly customizable, enabling users to easily add custom datasets, optical models, and spell checkers to suit specific requirements.

Furthermore, the project provides support for [MLflow tracking](https://mlflow.org/docs/latest/tracking.html#tracking), which enables better tracking and management of training and testing stages. MLflow allows users to log and compare experiments, track metrics, and store trained models for reproducibility and experimentation.

## Getting Started

To get started with the Graphite project, follow the steps below:

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:

```bash
git clone https://github.com/arthurflor23/handwritten-text-recognition.git
```

2. Navigate to the project directory:

```bash
cd handwritten-text-recognition
```

Create a virtual environment:

```bash
python3 -m venv venv
```

3. Activate the virtual environment:

* For Linux/Mac:

```bash
source venv/bin/activate
```

* For Windows:

```
venv\Scripts\activate
```

### Usage

The project offers various functionalities that can be accessed through command-line parameters. Below are some examples of how to use the project:

**Example 1: Perform optical model training**

```bash
python main.py --train --source iam --level line --network flor --epochs 1000 --batch-size 16
```

This command will train the optical model on the IAM dataset at the line level for 1000 epochs and batch size of 16.

**Example 2: Perform optical model testing**

```bash
python main.py --test --source iam --level line --network flor --beam-width 30 --top-paths 3 --run-index -1
```

This will perform testing on the IAM dataset at the line level using the Flor optical network architecture with a beam width of 30 and generating 3 top paths for prediction. The selected optical model is indicated by the run index, which loads the last trained one.

**Example 3: Perform optical model inference**

```bash
python main.py --infer --network flor --images path/to/image1.png path/to/image2.png --beam-width 30 --top-paths 3 --run-index -1
```

This will perform inferenceon the specified images using the Flor optical network architecture with a beam width of 30 and generating 3 top paths for prediction. The selected optical model is indicated by the run index, which loads the last trained one.

## Parameters

The Graphite project offers various command-line parameters that can be used to customize its behavior. Here is a list of the available parameters along with their descriptions:

- `--source`: Define the source data (e.g., iam, rimes).
- `--level`: Define the recognition level (e.g., line, paragraph).
- `--training-ratio`: Set the training partition ratio.
- `--validation-ratio`: Set the validation partition ratio.
- `--test-ratio`: Set the test partition ratio.
- `--lazy-mode`: Enable lazy loading.
- `--erosion`: Apply erosion augmentation (probability, kernel size, iterations).
- `--dilation`: Apply dilation augmentation (probability, kernel size, iterations).
- `--elastic-transform`: Apply elastic transform augmentation (probability, kernel size, alpha).
- `--perspective-transform`: Apply perspective transformation augmentation (probability, alpha).
- `--mixup`: Apply mixup augmentation (probability, opacity, iterations).
- `--shearing`: Apply shearing transformation augmentation (probability, angle).
- `--scaling`: Apply scaling transformation augmentation (probability, scale alpha).
- `--rotation`: Apply rotation transformation augmentation (probability, angle).
- `--translation`: Apply vertical and horizontal translation augmentation (probability, y-alpha, x-alpha).
- `--gaussian-noise`: Apply Gaussian noise augmentation (probability, alpha).
- `--gaussian-blur`: Apply Gaussian blur augmentation (probability, kernel size, iterations).
- `--disable-augmentation`: Disable data augmentation completely.
- `--network`: Define the optical network (e.g., bluche, flor, puigcerver).
- `--spell-checker`: Define the spell checker (e.g., openai).
- `--api-key`: Set the spell checker API key directly.
- `--env-key`: Define the environment variable that holds the API key.
- `--train`: Perform optical model training.
- `--epochs`: Number of epochs for training.
- `--batch-size`: Batch size for the generator.
- `--learning-rate`: Learning rate for the optimizer.
- `--plateau-factor`: Factor by which the learning rate will be reduced on a plateau.
- `--plateau-cooldown`: Cooldown period after a learning rate plateau is triggered.
- `--plateau-patience`: Number of epochs without improvement for the learning rate to be reduced.
- `--patience`: Number of epochs with no improvement after which training will be stopped.
- `--test`: Perform optical model test.
- `--beam-width`: The width of the beam for the CTC decoder.
- `--top-paths`: Number of top paths for prediction.
- `--share-top-paths`: Consider previous paths for the metrics.
- `--infer`: Perform inference process.
- `--images`: Set the image path list for handwriting recognition.
- `--bbox`: Set bounding box values (x, y, width, height).
- `--check`: Perform data verification.
- `--experiment-name`: Define MLflow experiment name.
- `--run-index`: Specify run index.
- `--seed`: Seed value for the training process.
- `--verbose`: Verbosity mode.

Feel free to experiment with the parameters to find the ones that best fit your specific needs.

## Call to Contribute

The Graphite is an ongoing project, and contributions are welcome. If you're interested in contributing to the project, here are a few ways you can get involved:

- **Bug Reports**: If you come across any bugs or issues while using the project, please report them on the [issue tracker](https://github.com/arthurflor23/handwritten-text-recognition/issues). Include as much detail as possible to help us reproduce and address the problem.

- **Feature Requests**: If you have any ideas or suggestions for new features or improvements, please submit them on the [issue tracker](https://github.com/arthurflor23/handwritten-text-recognition/issues). We appreciate your input and will review them for future development.

- **Pull Requests**: Contributions to the project are highly appreciated! If you would like to contribute code or any other improvements, please consider submitting a pull request.

- **Documentation**: Improving documentation is always valuable. If you notice any areas that can be clarified or expanded upon, please feel free to submit documentation updates or create new documentation pages.

- **Contact**: If you have any questions, need clarifications, or want to discuss any aspect of the project, feel free to get in touch. You can reach out via [email](mailto:afsn@ecomp.poli.br).

By contributing to Graphite, you'll be helping to improve the project for everyone. We appreciate your support and look forward to your contributions!

## Tutorial Notebook

To help you get started, a tutorial material has been created. This tutorial provides a step-by-step guide to exploring the end-to-end pipeline of the Handwritten Text Recognition project. 

The tutorial material is designed to be beginner-friendly and can be easily run on [Google Colab](https://research.google.com/colaboratory/), a cloud-based Jupyter notebook environment. It provides a hands-on experience of using the project's features and demonstrates the usage of various parameters and functionalities.

By following the tutorial, you will be able to:

- Understand the project's workflow and architecture.
- Learn how to set up the necessary dependencies and environment.
- Explore the different parameters and their descriptions.
- Perform tasks such as data training, and testing.
- Gain insights into the usage of data augmentation techniques.
- Get familiar with the optical network models and spell checkers available.
- Experience the integration with MLflow for experiment tracking.

To access the tutorial material, please refer to the [Tutorial Notebook](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/tutorial.ipynb) located in the project's repository. Follow the instructions within the notebook to run the code and explore the functionalities.

## Sponsor

This project is part of my PhD work and is currently under development. Your support would greatly contribute to its progress and success. If you find this project valuable or if it has helped you in any way, please consider showing your support by sponsoring it. Your sponsorship will help me dedicate more time and resources to enhance the project, implement new features, and conduct further research.

You can support this project through [Ko-fi](https://ko-fi.com/arthurflor23). Every contribution is greatly appreciated and motivates me to continue working on the project.

Thank you for considering sponsorship and for being a part of this journey!

## References

If you are interested in learning more about the project or the subject of Handwritten Text Recognition, you may find the following references:

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [HTR-Flor++: A Handwritten Text Recognition System Based on a Pipeline of Optical and Language Models.](https://doi.org/10.1145/3395027.3419603) Proceedings of the ACM Symposium on Document Engineering, 2020.

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. [Towards the Natural Language Processing as Spelling Correction for Offline Handwritten Text Recognition Systems.](https://doi.org/10.3390/app10217711) Applied Sciences, 2020.

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Lima, Estanislau B. and Toselli, Alejandro H. [HDSR-Flor: A Robust End-to-End System to Solve the Handwritten Digit String Recognition Problem in Real Complex Scenarios.](https://doi.org/10.1109/ACCESS.2020.3039003) IEEE Access, 2020.

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition.](https://doi.org/10.1109/SIBGRAPI51738.2020.00016) 33rd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), 2020.

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [A Robust Handwritten Recognition System for Learning on Different Data Restriction Scenarios.](https://doi.org/10.1016/j.patrec.2022.04.009) Pattern Recognition Letters, 2022.

These references provide additional insights and background information related to Handwritten Text Recognition and can be a valuable resource for further exploration. If any of these papers have been beneficial to your research or project, it would be greatly appreciated if you could consider citing them.
