<img src="https://github.com/arthurflor23/handwritten-text-recognition/blob/master/doc/images/000.png?raw=true">

Handwritten Text Recognition (HTR) system implemented using TensorFlow 2.0 and trained on the Bentham/IAM/Rimes/Saint Gall offline HTR datasets. This Neural Network model recognizes the text contained in the images of segmented texts lines.

Data partitioning (train, validation, test) was performed following the methodology of each dataset. The project implemented the HTRModel abstraction model (adapted from the [CTCModel](https://github.com/ysoullard/CTCModel)) as a way to facilitate the development of HTR systems.

**Notes**:
1. All **references** are commented in the code.
2. This project **doesn't offer** post-processing, such as N-gram Language Model.
3. Some results (txt files reports) can be find in **doc** folder, divided by dataset/architecture.
4. For more information and demo run step by step, check out the **[tutorial](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.

## Datasets supported

a. [Bentham](http://transcriptorium.eu/datasets/bentham-collection/)

b. [IAM](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database)

c. [Rimes](http://www.a2ialab.com/doku.php?id=rimes_database:start)

d. [Saint Gall](http://www.fki.inf.unibe.ch/databases/iam-historical-document-database/saint-gall-database)

## Requirements

* Python 3.x
* OpenCV 4.x
* editdistance
* TensorFlow 2.0

## Command line arguments

* `--dataset`: dataset name (bentham, iam, rimes, saintgall)
* `--arch`: network to be used (puigcerver, bluche, flor)
* `--transform`: transform dataset to the HDF5 file
* `--cv2`: visualize sample from transformed dataset
* `--train`: train model using the dataset argument
* `--test`: evaluate and predict model using the dataset argument
* `--epochs`: number of epochs
* `--batch_size`: number of the size of each batch

## Tutorial (Google Colab/Drive)

A Jupyter Notebook is available to demo run, check out the **[tutorial](https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/tutorial.ipynb)** on Google Colab/Drive.