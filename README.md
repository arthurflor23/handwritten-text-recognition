<img src="https://github.com/arthurflor23/handwritten-text-recognition/blob/master/doc/image/header.png?raw=true">

Handwritten Text Recognition (HTR) system implemented using [TensorFlow 2.x](https://www.tensorflow.org/) and trained on the Bentham/IAM/Rimes/Saint Gall/Washington offline HTR datasets. This Neural Network model recognizes the text contained in the images of segmented texts lines.

Data partitioning (train, validation, test) was performed following the methodology of each dataset. The project implemented the HTRModel abstraction model (inspired by [CTCModel](https://github.com/ysoullard/CTCModel)) as a way to facilitate the development of HTR systems.

**Notes**:

1. All **references** are commented in the code.
2. This project **doesn't offer** post-processing, such as Statistical Language Model.
3. Check out the presentation in the **doc** folder.
4. For more information and demo run step by step, check out the **[tutorial](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.

## Datasets supported

a. [Bentham](http://transcriptorium.eu/datasets/bentham-collection/)

b. [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

c. [Rimes](http://www.a2ialab.com/doku.php?id=rimes_database:start)

d. [Saint Gall](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/saint-gall-database)

e. [Washington](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/washington-database)

## Requirements

- Python 3.x
- OpenCV 4.x
- editdistance
- TensorFlow 2.x

## Command line arguments

- `--source`: dataset/model name (bentham, iam, rimes, saintgall, washington)
- `--arch`: network to be used (puigcerver, bluche, flor)
- `--transform`: transform dataset to the HDF5 file
- `--cv2`: visualize sample from transformed dataset
- `--kaldi_assets`: save all assets for use with kaldi
- `--image`: predict a single image with the source parameter
- `--train`: train model using the source argument
- `--test`: evaluate and predict model using the source argument
- `--norm_accentuation`: discard accentuation marks in the evaluation
- `--norm_punctuation`: discard punctuation marks in the evaluation
- `--epochs`: number of epochs
- `--batch_size`: number of the size of each batch

## Tutorial (Google Colab/Drive)

A Jupyter Notebook is available to demo run, check out the **[tutorial](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.

## Sample

Bentham sample with default parameters in the **[tutorial](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/tutorial.ipynb)** file.

1. Preprocessed image (network input)
2. TE_L: Ground Truth Text (label)
3. TE_P: Predicted text (network output)

<img src="https://github.com/arthurflor23/handwritten-text-recognition/blob/master/doc/image/bentham_sample.png?raw=true">

## Citation

If this project helped in any way in your research work, feel free to cite the following papers:

```
@inproceedings{10.1145/3395027.3419603,
    author      = {Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H. and Lima, Estanislau B.},
    title       = {{HTR-Flor++:} A Handwritten Text Recognition System Based on a Pipeline of Optical and Language Models},
    booktitle   = {Proceedings of the ACM Symposium on Document Engineering 2020},
    year        = {2020},
    publisher   = {Association for Computing Machinery},
    address     = {New York, NY, USA},
    location    = {Virtual Event, CA, USA},
    series      = {DocEng '20},
    isbn        = {9781450380003},
    url         = {https://doi.org/10.1145/3395027.3419603},
    doi         = {10.1145/3395027.3419603},
}
```

```
@article{10.3390/app10217711,
    author  = {Neto, Arthur F. S. and Bezerra, Byron L. D. and Toselli, Alejandro H.},
    title   = {Towards the Natural Language Processing as Spelling Correction for Offline Handwritten Text Recognition Systems},
    journal = {Applied Sciences},
    pages   = {1-29},
    month   = {10},
    year    = {2020},
    volume  = {10},
    number  = {21},
    url     = {https://doi.org/10.3390/app10217711},
    doi     = {10.3390/app10217711},
}
```