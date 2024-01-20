# Handwritten Text Recognition with Graphite

This project aims to provide a comprehensive solution for Handwritten Text Recognition (HTR) using [Tensorflow](https://www.tensorflow.org/). It includes a tutorial and a set of tools for data processing, model training, testing, and inference. The HTR model can be trained on various datasets and supports different levels of recognition, such as line and paragraph level. The project also supports generative and language models that make up the workflow for handwriting synthesis and spelling correction.

Furthermore, the project provides support for [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html#tracking), which enables better tracking and management of training and testing phases. MLflow allows you to log and compare experiments, track metrics, and store trained models for reproducibility. Explore the [MLflow Dashboard](https://mlflow.org/docs/latest/tracking.html#explore-runs-and-results) and track experiments with `mlflow ui`.

## Getting Started

To get started with the Graphite project, follow the steps below.

### Prerequisites

-   Python 3.9 or higher
-   pip package manager

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

-   For Linux/Mac:

```bash
source .venv/bin/activate
```

-   For Windows:

```
.venv\Scripts\activate
```

4. Install requirements

```
pip install -r requirements.txt
```

### Datasets

The project supports a wide range of datasets for handwritten text recognition. The following datasets are already integrated into the project and can be easily used for training and evaluation.

1. [Bentham](https://drive.google.com/file/d/1JTF7itNMavb81EUTDRkUAUbsJ1wQx_Mw/view?usp=drive_link) [[1]](https://doi.org/10.1109/DAS.2014.23)

2. [CVL-Digits](https://drive.google.com/file/d/1s_13CvgVtYYMynL1wv1o7eI4QOkDgaGW/view?usp=drive_link) [[2]](https://doi.org/10.1109/ICFHR.2014.136)

3. [IAM](https://drive.google.com/file/d/1-_tFR64Tko7ICnZMIdwUQraXAYd028R2/view?usp=drive_link) [[3]](https://doi.org/10.1007/s100320200071)

4. [ORAND-CAR](https://drive.google.com/file/d/1NWywg7urSy-7asGFCfRrMuEugh59E6K1/view?usp=drive_link) [[2]](https://doi.org/10.1109/ICFHR.2014.136)

5. [Parzival](https://drive.google.com/file/d/1EYv33klUY_tnLMbZ07I_upEdTZAkv_58/view?usp=drive_link) [[3]](http://dx.doi.org/10.1109/ICDAR.2011.20)

6. [Rimes](https://drive.google.com/file/d/1Hjnc_cYzzPg88382pvP04skjsq6acQGa/view?usp=drive_link) [[4]](https://hal.science/hal-01395332)

7. [Saint Gall](https://drive.google.com/file/d/1o3hvTBcr-h6S45mT9vDWKRokFN9Rqfis/view?usp=drive_link) [[5]](https://dl.acm.org/doi/10.1145/2037342.2037348)

8. [Washington](https://drive.google.com/file/d/150I0IcSWsjikYARuBBqmjrRQwukwSzRh/view?usp=drive_link) [[6]](http://dx.doi.org/10.1016/j.patrec.2011.09.009)

## Parameters

The project has several command-line parameters that can be used to customize its behavior. The list of available parameters is outlined below, along with their descriptions.

#### Models

-   `--synthesis`: Define the handwriting synthesis model to be used (e.g., gan).
-   `--recognition`: Define the recognition model to be used (e.g., bluche, flor, puigcerver).
-   `--spelling`: Define the spelling model to be used (e.g., openai).

#### MLflow

-   `--synthesis-run-index`: Define a synthesis model run index.
-   `--recognition-run-index`: Define a recognition model run index.
-   `--experiment-name`: Define a MLflow experiment name.

#### Dataset

-   `--source`: Define the source data (e.g., iam, rimes).
-   `--source-input-path`: Source input path.
-   `--text-level`: Define the text structure level (e.g., line, paragraph).
-   `--image-shape`: Define the image shape (width, height, channels).
-   `--char-width`: Define character width for normalization.
-   `--training-ratio`: Set the training partition ratio.
-   `--validation-ratio`: Set the validation partition ratio.
-   `--test-ratio`: Set the test partition ratio.
-   `--lazy-mode`: Enable lazy loading mode.

#### Augmentor

-   `--binarize`: Apply binarization thresholding (probability, method name).
-   `--erode`: Apply erode transformation (probability, kernel size, iterations).
-   `--dilate`: Apply dilate transformation (probability, kernel size, iterations).
-   `--elastic`: Apply elastic transformation (probability, kernel size, alpha).
-   `--perspective`: Apply perspective transformation (probability, alpha).
-   `--mixup`: Apply mixup transformation (probability, opacity, iterations).
-   `--shear`: Apply shear transformation (probability, angle).
-   `--scale`: Apply scale transformation (probability, scale alpha).
-   `--rotate`: Apply rotate transformation (probability, angle).
-   `--shift-y`: Apply vertical translation (probability, y-alpha).
-   `--shift-x`: Apply horizontal translation (probability, x-alpha).
-   `--salt-and-pepper`: Apply Salt and Pepper noise (probability, alpha).
-   `--gaussian-noise`: Apply Gaussian noise (probability, alpha).
-   `--gaussian-blur`: Apply Gaussian blur (probability, kernel size, iterations).
-   `--disable-augmentation`: Disable data augmentation completely.

#### Synthesis

-   `--discriminator-steps`: Set the repetition of steps for discriminator training in synthesis workflow.
-   `--generator-steps`: Set the skipping steps for generator training in synthesis workflow.
-   `--synthesis-ratio`: Set the synthetic data ratio for synthesis and recognition workflow.

#### Training

-   `--training`: Perform training pipeline.
-   `--epochs`: Maximum number of epochs.
-   `--batch-size`: Batch size for the generator.
-   `--learning-rate`: Learning rate for the optimizer.
-   `--plateau-factor`: Factor for reducing the learning rate.
-   `--plateau-cooldown`: Epochs to wait after a learning rate reduction.
-   `--plateau-patience`: Epochs without improvement before reducing the learning rate.
-   `--patience`: Epochs without improvement before stopping training.

#### Test

-   `--test`: Perform test pipeline.
-   `--top-paths`: Number of top paths for prediction.
-   `--beam-width`: Beam width for CTC decoder.

#### Inference

-   `--inference`: Perform inference pipeline.
-   `--inference-output-path`: Inference output path.
-   `--image`: Image path for handwriting recognition.
-   `--bbox`: Bounding box values for image (x, y, width, height).
-   `--text`: Text input for handwriting synthesis.

#### Others

-   `--check`: Perform check pipeline.
-   `--seed`: Seed value.

### Usage

The project offers a range of functionalities through command-line parameters; feel free to experiment with these to find the ones that best suit your specific needs. Below are some examples of usage.

**Example 1: Perform recognition model training**

```bash
python graphite --source iam --text-level line --recognition flor --batch-size 16 --training
```

This command will train the recognition model on IAM dataset at the line level, using the Flor optical network with batch size of 16.

**Example 2: Perform recognition model testing**

```bash
python graphite --source iam --text-level line --recognition flor --beam-width 30 --top-paths 3 --recognition-run-index -1 --test
```

This command will perform testing phase on IAM dataset using the Flor optical network and a beam width of 30 with 3 top paths in the prediction. The selected optical model is indicated by the recognition run index, which loads the last trained model.

**Example 3: Perform recognition model inference**

```bash
python graphite --recognition flor --beam-width 30 --recognition-run-index -1 --inference --image path/to/image1.png
```

This command will perform inference on the specified images using the Flor optical network and a beam width of 30 in the prediction. The selected optical model is indicated by the recognition run index, which loads the last trained model.

---

Additionally, different workflows can be used, such as `--synthesis` and the combination of `--synthesis` with `--recognition`. For the first, the synthesis model is trained and used to synthesize fake manuscripts; in the second, the synthesis serves as data augmentation for the recognition models in an integrated training pipeline.

## Tutorial Notebook

To help you get started, a tutorial material has been created. This tutorial provides a step-by-step guide to exploring the main pipeline of the project.

The tutorial is designed to be beginner-friendly and can be easily run on [Google Colab](https://research.google.com/colaboratory/), a cloud-based Jupyter notebook environment. It provides a hands-on experience of using the project's features and demonstrates the usage of various parameters and functionalities.

By following the tutorial, you'll be able to:

-   Understand the project's pipeline.
-   Learn how to set up required dependencies and environment.
-   Explore different parameters.
-   Execute data training and testing pipelines.
-   Gain insights into your own context problem.

To access the material, see the [Tutorial Jupyter Notebook](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/tutorial.ipynb) located in the project repository. Follow the notebook instructions to run the code and explore the features.

## Sponsor

This project is part of the PhD work and is currently in parallel development. Thus, your support would greatly contribute to its progress. If you find this project valuable or if it has helped you in any way, please consider showing your support by sponsoring it. The sponsorship will help me dedicate more time and resources to enhance the project and implement new features.

You can support this project through [Ko-fi](https://ko-fi.com/arthurflor23). Thank you for considering sponsorship.

## References

If you are interested in learning more about the project or the subject of Handwritten Text Recognition, you may be interested in the following references:

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition.](https://doi.org/10.1109/SIBGRAPI51738.2020.00016) 33rd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), 2020.

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [HTR-Flor++: A Handwritten Text Recognition System Based on a Pipeline of Optical and Language Models.](https://doi.org/10.1145/3395027.3419603) Proceedings of the ACM Symposium on Document Engineering, 2020.

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Lima, Estanislau B. and Toselli, Alejandro H. [HDSR-Flor: A Robust End-to-End System to Solve the Handwritten Digit String Recognition Problem in Real Complex Scenarios.](https://doi.org/10.1109/ACCESS.2020.3039003) IEEE Access, 2020.

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. [Towards the Natural Language Processing as Spelling Correction for Offline Handwritten Text Recognition Systems.](https://doi.org/10.3390/app10217711) Applied Sciences, 2020.

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [A Robust Handwritten Recognition System for Learning on Different Data Restriction Scenarios.](https://doi.org/10.1016/j.patrec.2022.04.009) Pattern Recognition Letters, 2022.

These references provide additional insights and background information related to Handwritten Text Recognition and can be a valuable resource for further exploration. If any of these papers have been beneficial to your research or project, it would be greatly appreciated if you could consider citing them.
