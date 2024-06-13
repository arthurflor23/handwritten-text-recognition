<img src="https://github.com/arthurflor23/handwritten-text-recognition/blob/master/doc/image/header.png?raw=true">

Handwritten Text Recognition (HTR) system implemented using [TensorFlow 2.x](https://www.tensorflow.org/) and trained on the Bentham/IAM/Rimes/Saint Gall/Washington offline HTR datasets. This Neural Network model recognizes the text contained in the images of segmented texts lines.

Data partitioning (train, validation, test) was performed following the methodology of each dataset. The project implemented the HTRModel abstraction model (inspired by [CTCModel](https://github.com/ysoullard/CTCModel)) as a way to facilitate the development of HTR systems.

**Notes**:

1. All **references** are commented in the code.
2. This project **doesn't offer** post-processing, such as Statistical Language Model.
3. Check out the presentation in the **doc** folder.
4. For more information and demo run step by step, check out the **[tutorial](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.

## Datasets supported

a. [Bentham](http://www.transcriptorium.eu/~tsdata/)

b. [BRESSAY](https://zenodo.org/records/11637681)

c. [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

d. [Rimes](http://www.a2ialab.com/doku.php?id=rimes_database:start)

e. [Saint Gall](https://fki.tic.heia-fr.ch/databases/saint-gall-database)

f. [Washington](https://fki.tic.heia-fr.ch/databases/washington-database)

## Requirements

-   Python 3.x
-   OpenCV 4.x
-   editdistance
-   TensorFlow 2.x

## Command line arguments

-   `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
-   `--arch`: network to be used (puigcerver, bluche, flor)
-   `--transform`: transform dataset to the HDF5 file
-   `--cv2`: visualize sample from transformed dataset
-   `--kaldi_assets`: save all assets for use with kaldi
-   `--image`: predict a single image with the source parameter
-   `--train`: train model using the source argument
-   `--test`: evaluate and predict model using the source argument
-   `--norm_accentuation`: discard accentuation marks in the evaluation
-   `--norm_punctuation`: discard punctuation marks in the evaluation
-   `--epochs`: number of epochs
-   `--batch_size`: number of the size of each batch

## Tutorial (Google Colab/Drive)

A Jupyter Notebook is available to demo run, check out the **[tutorial](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.

## Sample

Bentham sample with default parameters in the **[tutorial](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/tutorial.ipynb)** file.

1. Preprocessed image (network input)
2. TE_L: Ground Truth Text (label)
3. TE_P: Predicted text (network output)

<img src="https://github.com/arthurflor23/handwritten-text-recognition/blob/master/doc/image/bentham_sample.png?raw=true">

## References

If you are interested in learning more about the project or the subject of Handwritten Text Recognition, you may be interested in the following references:

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [HTR-Flor: A Deep Learning System for Offline Handwritten Text Recognition.](https://doi.org/10.1109/SIBGRAPI51738.2020.00016) 33rd SIBGRAPI Conference on Graphics, Patterns and Images (SIBGRAPI), 2020.

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [HTR-Flor++: A Handwritten Text Recognition System Based on a Pipeline of Optical and Language Models.](https://doi.org/10.1145/3395027.3419603) Proceedings of the ACM Symposium on Document Engineering, 2020.

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Lima, Estanislau B. and Toselli, Alejandro H. [HDSR-Flor: A Robust End-to-End System to Solve the Handwritten Digit String Recognition Problem in Real Complex Scenarios.](https://doi.org/10.1109/ACCESS.2020.3039003) IEEE Access, 2020.

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. [Towards the Natural Language Processing as Spelling Correction for Offline Handwritten Text Recognition Systems.](https://doi.org/10.3390/app10217711) Applied Sciences, 2020.

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B. [A Robust Handwritten Recognition System for Learning on Different Data Restriction Scenarios.](https://doi.org/10.1016/j.patrec.2022.04.009) Pattern Recognition Letters, 2022.

-   Neto, Arthur F. S. and Bezerra, Byron L. D. and Moura, Gabriel C. D. and Toselli, Alejandro H. [Data Augmentation for Offline Handwritten Text Recognition: A Systematic Literature Review](https://doi.org/10.1007/s42979-023-02583-6). SN Computer Science, 2024.

-   Neto, A. F. S., Bezerra, B. L. D., Araujo, S. S., Souza, W. M. A. S., Alves, K. F., Oliveira, M. F., Lins, S. V. S., Hazin, H. J. F., Rocha, P. H. V., Toselli, A. H.: [BRESSAY: A Brazilian Portuguese Dataset for Offline Handwritten Text Recognition](https://icdar2024.net/). In: 18th International Conference on Document Analysis and Recognition (ICDAR). Springer, Athens, Greece (9 2024).
