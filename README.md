# Handwritten Text Synthesis and Recognition (pySarah)

The `pySarah` project provides a solution for Handwritten Text Recognition (HTR) using [Tensorflow](https://www.tensorflow.org/). It includes a tutorial and a set of tools for data processing, model training, testing, and inference. The HTR model can be trained on various datasets and supports different levels of recognition, such as line and paragraph level. The project also supports generative and language models that make up the workflow for handwriting synthesis and spelling correction.

The project provides support for [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html#tracking), which enables better tracking and management of training and testing phases. MLflow allows you to log and compare experiments, track metrics, and store trained models for reproducibility. Explore the [MLflow Dashboard](https://mlflow.org/docs/latest/tracking.html#explore-runs-and-results) and track experiments with `mlflow ui`.

## Getting Started

To get started with the project, follow the steps below.

### Prerequisites

- Python 3.11 or higher
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

3. Create and activate the virtual environment:

```bash
python3 -m venv .venv
```

- For Linux/Mac:

```bash
source .venv/bin/activate
```

- For Windows:

```
.venv\Scripts\activate
```

4. Install requirements

```
pip install -r requirements.txt
```

### Datasets

The project supports a wide range of datasets for handwritten text recognition. The following datasets are already integrated into the project and can be easily used for training and evaluation.

- [Bentham](https://drive.google.com/file/d/1do3tS7vd-QaUxkeBwPE4Phia99-J_AUq/view?usp=sharing) [[paper]](https://doi.org/10.1109/DAS.2014.23)

- [BRESSAY](https://zenodo.org/records/11637681) [[paper]](https://doi.org/10.1007/978-3-031-70536-6_19)

- [CVL-Database](https://drive.google.com/file/d/1H0M2lHdxUCLs7eKjk2q7fNBYU0-PkLDo/view?usp=sharing) [[paper]](https://doi.org/10.1109/ICDAR.2013.117)

- [CVL-Digits](https://drive.google.com/file/d/1jPZtUtiARgrbFaCfPyPg_UZrgwYHJ7Td/view?usp=sharing) [[paper]](https://doi.org/10.1109/ICFHR.2014.136)

- [EMNIST](https://drive.google.com/file/d/107Lzd-5fAkt3XLdOgTWH5yQeqd3QHbKf/view?usp=sharing) [[paper]](https://doi.org/10.1109/ijcnn.2017.7966217)

- [IAM](https://drive.google.com/file/d/1z6gOT4U_eTsCguSCWAz3IaQwTD8TXwDw/view?usp=sharing) [[paper]](https://doi.org/10.1007/s100320200071)

- [MNIST](https://drive.google.com/file/d/1QwW7RgxQjJ83nJQ5dMw4mWawyCcpwRp8/view?usp=sharing) [[paper]](http://yann.lecun.com/exdb/mnist/)

- [ORAND-CAR](https://drive.google.com/file/d/1jkT2ow85eob9hK4xygdOjlIT6zNMUlAG/view?usp=sharing) [[paper]](https://doi.org/10.1109/ICFHR.2014.136)

- [Parzival](https://drive.google.com/file/d/1szhdRxYRCkkIehQaLgTk3YD9At563BSU/view?usp=sharing) [[paper]](http://dx.doi.org/10.1109/ICDAR.2011.20)

- [RIMES](https://drive.google.com/file/d/1iax6qNqKtg0PHZl68HnJy2WRFQ2IcRLM/view?usp=sharing) [[paper]](https://hal.science/hal-01395332)

- [Saint Gall](https://drive.google.com/file/d/1X66lsJFEK-RixO4dQ0DP9ZH6yVc64P2m/view?usp=sharing) [[paper]](https://dl.acm.org/doi/10.1145/2037342.2037348)

- [Washington](https://drive.google.com/file/d/1MuKc3D3SoWVUJPYqhmnOT9Xd-CWnmpk4/view?usp=sharing) [[paper]](http://dx.doi.org/10.1016/j.patrec.2011.09.009)

## Parameters

The project has several command-line parameters that can be used to customize its behavior. The list of available parameters is outlined below, along with their descriptions.

#### Models

- `--synthesis`: Specify synthesis model (e.g., flor).
- `--recognition`: Specify recognition model (e.g., flor).
- `--segmentation`: Specify segmentation model (e.g., flor).
- `--writer-identification`: Specify writer identification model (e.g., flor).
- `--spelling`: Specify spelling model (e.g., openai).

#### MLflow

- `--synthesis-run-id`: Synthesis model run id or index.
- `--recognition-run-id`: Recognition model run id or index.
- `--segmentation-run-id`: Segmentation model run id or index.
- `--writer-identification-run-id`: Writer identification model run id or index.
- `--experiment-name`: MLflow experiment name.
- `--finished-runs`: Only finished runs for selection.

#### Dataset

- `--source`: Source data (e.g., iam).
- `--text-level`: Text structure level (e.g., line).
- `--image-shape`: Image dimensions (height, width, channels).
- `--char-width`: Character width for normalization.
- `--mask-by-text`: Mask data by text length.
- `--order-by-text`: Sort data by text length.
- `--training-ratio`: Training partition ratio.
- `--validation-ratio`: Validation partition ratio.
- `--test-ratio`: Test partition ratio.
- `--illumination` : Apply illumination compensation.
- `--binarization` : Apply binarization method.
- `--lazy-mode`: Activate lazy loading.

#### Augmentor

- `--mixup`: Mixup transformation (probability, opacity, iterations).
- `--erode`: Erode transformation (probability, kernel size, iterations).
- `--dilate`: Dilate transformation (probability, kernel size, iterations).
- `--elastic`: Elastic transformation (probability, kernel size, alpha).
- `--perspective`: Perspective transformation (probability, alpha).
- `--shear`: Shear transformation (probability, alpha).
- `--rotate`: Rotate transformation (probability, alpha).
- `--scale`: Scale transformation (probability, alpha).
- `--shift-y`: Vertical translation (probability, alpha).
- `--shift-x`: Horizontal translation (probability, alpha).
- `--salt-and-pepper`: Salt and Pepper noise (probability, alpha).
- `--gaussian-noise`: Gaussian noise (probability, alpha).
- `--gaussian-blur`: Gaussian blur filter (probability, kernel size).
- `--skip-augmentation`: Skip data augmentation.

#### Synthesis

- `--discriminator-steps`: Repetition of steps for discriminator training in synthesis workflow.
- `--generator-steps`: Skipping steps for generator training in synthesis workflow.

#### Training

- `--training`: Perform training pipeline.
- `--training-step-factor`: Factor for training steps.
- `--epochs`: Maximum number of epochs.
- `--batch-size`: Batch size.
- `--learning-rate`: Learning rate.
- `--plateau-factor`: Learning rate reduction factor.
- `--plateau-cooldown`: Cooldown after plateau.
- `--plateau-patience`: Plateau patience epochs.
- `--patience`: Stop after no improvement.
- `--synthesis-probability`: Training with synthetic data.

#### Test

- `--test`: Perform test pipeline.
- `--top-paths`: Top paths for prediction.
- `--beam-width`: CTC decoder beam width.

#### Inference

- `--inference`: Perform inference pipeline.
- `--image`: Image path for recognition.
- `--bbox`: Bounding box (x, y, width, height).
- `--text`: Text for synthesis.

#### Others

- `--check`: Perform check pipeline.
- `--input-path`: Path to input data.
- `--output-path`: Path to output data.
- `--gpu`: GPU index or sequence of indices.
- `--seed`: Seed value.
- `--verbose`: Verbosity level.

### Usage

The project offers a range of functionalities through command-line parameters; feel free to experiment with these to find the ones that best suit your specific needs. Below are some examples of usage.

**Example 1: Perform recognition model training**

```bash
python sarah --source iam --text-level line --recognition flor --batch-size 8 --training
```

This command will train the recognition model on IAM dataset at the line level, using the Flor optical network with batch size of 8.

**Example 2: Perform recognition model testing**

```bash
python sarah --source iam --text-level line --recognition flor --beam-width 33 --recognition-run-id -1 --test
```

This command will perform testing phase on IAM dataset using the Flor optical network and a beam width of 32. The selected optical model is indicated by the recognition run id, which loads the last trained model.

**Example 3: Perform recognition model inference**

```bash
python sarah --recognition flor --recognition-run-id -1 --inference --image path/to/image1.png
```

This command will perform inference on the specified images using the Flor optical network. The selected optical model is indicated by the recognition run id, which loads the last trained model.

---

Additionally, different workflows can be used, such as `--synthesis` and the combination of `--synthesis` with `--recognition`. For the first, the synthesis model is trained and used to synthesize fake manuscripts; in the second, the synthesis serves as data augmentation for the recognition models in an integrated training pipeline.

## Tutorial Notebook

To help you get started, a tutorial material has been created. This tutorial provides a step-by-step guide to exploring the main pipeline of the project.

The tutorial is designed to be beginner-friendly and can be easily run on [Google Colab](https://research.google.com/colaboratory/), a cloud-based Jupyter notebook environment. It provides a hands-on experience of using the project's features and demonstrates the usage of various parameters and functionalities.

By following the tutorial, you'll be able to:

- Understand the project's pipeline.
- Learn how to set up required dependencies and environment.
- Explore different parameters.
- Execute data training and testing pipelines.
- Gain insights into your own context problem.

To access the material, see the [Tutorial Jupyter Notebook](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/tutorial.ipynb) located in the project repository. Follow the notebook instructions to run the code and explore the features.

## Sponsor

This project is part of the PhD work and is currently in parallel development. Thus, your support would greatly contribute to its progress. If you find this project valuable or if it has helped you in any way, please consider showing your support by sponsoring it. The sponsorship will help me dedicate more time and resources to enhance the project and implement new features.

You can support this project through [Ko-fi](https://ko-fi.com/arthurflor23). Thank you for considering sponsorship.

## References

If you are interested in learning more about the project or the subject of Handwritten Text Recognition, you may be interested in the following references:

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition](https://doi.org/10.1109/SIBGRAPI51738.2020.00016). 33rd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), 2020.

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [HTR-Flor++: A Handwritten Text Recognition System Based on a Pipeline of Optical and Language Models](https://doi.org/10.1145/3395027.3419603). Proceedings of the ACM Symposium on Document Engineering, 2020.

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Lima, Estanislau B. and Toselli, Alejandro H. [HDSR-Flor: A Robust End-to-End System to Solve the Handwritten Digit String Recognition Problem in Real Complex Scenarios](https://doi.org/10.1109/ACCESS.2020.3039003). IEEE Access, 2020.

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. [Towards the Natural Language Processing as Spelling Correction for Offline Handwritten Text Recognition Systems](https://doi.org/10.3390/app10217711). Applied Sciences, 2020.

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [A Robust Handwritten Recognition System for Learning on Different Data Restriction Scenarios](https://doi.org/10.1016/j.patrec.2022.04.009). Pattern Recognition Letters, 2022.

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Moura, Gabriel C. D. and Toselli, Alejandro H. [Data Augmentation for Offline Handwritten Text Recognition: A Systematic Literature Review](https://doi.org/10.1007/s42979-023-02583-6). SN Computer Science, 2024.

- Neto, A. F. S., Bezerra, B. L. D., Araujo, S. S., Souza, W. M. A. S., Alves, K. F., Oliveira, M. F., Lins, S. V. S., Hazin, H. J. F., Rocha, P. H. V., Toselli, A. H.: [BRESSAY: A Brazilian Portuguese Dataset for Offline Handwritten Text Recognition](https://doi.org/10.1007/978-3-031-70536-6_19). In: 18th International Conference on Document Analysis and Recognition (ICDAR). Springer, Athens, Greece (9 2024).

- Neto, A. F. S., Bezerra, B. L. D., Araujo, S. S., Souza, W. M. A. S., Alves, K. F., Oliveira, M. F., Lins, S. V. S., Hazin, H. J. F., Rocha, P. H. V., Toselli, A. H.: [ICDAR 2024 Competition on Handwritten Text Recognition in Brazilian Essays – BRESSAY](https://doi.org/10.1007/978-3-031-70552-6_21). In: 18th International Conference on Document Analysis and Recognition (ICDAR). Springer, Athens, Greece (9 2024).

- Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. [HTSR-Pollen: Handwritten Text Synthesis and Recognition System to Overcome Data Scarcity](https://doi.org/10.1109/ACCESS.2026.3681630). IEEE Access, 2026.

These references provide additional insights and background information related to Handwritten Text Recognition and can be a valuable resource for further exploration. If any of these papers have been beneficial to your research or project, it would be greatly appreciated if you could consider citing them.
