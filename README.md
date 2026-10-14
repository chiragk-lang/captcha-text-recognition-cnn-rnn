# Captcha Text Recognition using Custom CNN-RNN Models

## Overview

This project focuses on recognizing six-character CAPTCHA text from images using custom deep learning architectures built with PyTorch.

Three CNN-RNN models are trained and evaluated individually. Their predicted character probabilities are then combined using a simple ensemble to improve recognition performance.

The project covers image preprocessing, data augmentation, character encoding, model development, training, validation, evaluation, and test-set prediction generation.

## Models Implemented

* **CRNNFixed:** A convolutional neural network (CNN) for visual feature extraction, followed by a bidirectional LSTM for sequence modeling.
* **AttentionCRNN:** A CNN-BiLSTM architecture with a simple attention mechanism.
* **GRUCRNN:** A CNN followed by a bidirectional GRU for sequence modeling.
* **Three-model ensemble:** Averages the output probability distributions of the three models before selecting the predicted characters.

No pretrained model weights are used.

## Methodology

1. Load the labeled CAPTCHA images and their corresponding text.
2. Inspect the dataset and correct two labels after manually checking their images.
3. Map the character vocabulary to numerical indices.
4. Split the labeled data into training and validation sets using an 80:20 split.
5. Apply grayscale conversion, normalization, and data augmentation to training images.
6. Train the custom CNN-RNN models.
7. Evaluate the models using character accuracy, full-sequence accuracy, and character error rate (CER).
8. Combine the three models using probability averaging.
9. Generate predictions for the test images and export them as a CSV file.

## Dataset

The dataset contains labeled CAPTCHA images for training and a separate set of images for prediction.

* Training images: 20,000
* Validation images: 4,000
* Test images: 5,000
* Number of unique characters: 31

The labeled data is divided into training and validation subsets using an 80:20 split.

**Note:** The dataset itself is not included in this repository. Obtain the required files from the original competition or dataset source, and follow its licensing and sharing rules.

The notebook expects the dataset archive and directory structure to match its data-loading code. In particular, the code uses `cig_ps/train_images`, `cig_ps/test_images`, and `cig_ps/train-labels.csv` under the configured runtime path.

## Validation Results

The recorded validation results were:

| Model                | Validation Character Error Rate |
| -------------------- | ------------------------------: |
| CRNNFixed            |                        0.007000 |
| GRUCRNN              |                        0.010667 |
| AttentionCRNN        |                        0.016333 |
| Three-model ensemble |                        0.001667 |

The ensemble achieved the lowest recorded validation CER among the evaluated approaches.

**Metric:** Character error rate (CER) measures the total edit distance between predicted and reference text, divided by the total number of reference characters. Lower values indicate fewer character-level errors.

These are validation results from the notebook, not independently verified test-set performance.

## Technologies Used

* Python
* PyTorch
* NumPy
* Pandas
* Matplotlib
* Pillow (PIL)
* scikit-learn
* torchvision

## Repository Contents

* `notebook_Chirag_Kumar_24117040.ipynb` — notebook containing preprocessing, model implementations, training, evaluation, ensemble inference, and CSV generation.
* `submission_ChiragKumar_24117040.csv` — generated test-set predictions, included only if the competition rules permit sharing them.
* `README.md` — project overview, methodology, results, and setup notes.

## How to Run

1. Open the notebook in Google Colab or another compatible Python environment.
2. Install the required dependencies if they are not already available.
3. Obtain the dataset from its authorized source.
4. Place the dataset archive and extracted files in the locations expected by the notebook. Update the paths if your runtime uses a different directory.
5. Run the notebook cells in order to preprocess the images, train or load the models, evaluate the ensemble, and generate predictions.
6. The final prediction CSV is saved as `submission_ChiragKumar_24117040.csv`.

Training from scratch can take significant time and benefits from a GPU. The notebook also checks for saved model checkpoints; these are not included in this repository by default.

## Future Improvements

* Experiment with alternative CNN feature extractors and sequence models.
* Explore CTC-based sequence decoding for variable-length text recognition.
* Compare ensemble strategies and improve inference efficiency.
* Add a standalone inference script for predicting text from individual CAPTCHA images.

## Author

Chirag Kumar

