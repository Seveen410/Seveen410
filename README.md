# Speech Command Recognition 

## Project Objective

This project aims to classify audio data from the **Google Speech Commands** dataset into predefined spoken commands . The model is designed to recognize speech commands such as "yes", "no", "up", "down", and more, by analyzing the spectrograms of the audio files.

### Objectives:
- Build a model for audio classification.
- **Preprocess audio data** .
- **Evaluate the model** using various metrics such as accuracy and confusion matrix.

---

## Dataset Information and Preprocessing Steps

### Dataset

This project utilizes the **Google Speech Commands** dataset, which contains audio clips of various speech commands spoken by different individuals. The dataset includes two types of words:

#### Command words:
- "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go", "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine".

#### Auxiliary words:
- "bed", "bird", "cat", "dog", "happy", "house", "marvin", "sheila", "tree", "wow".

### Preprocessing

- **Loading the data**: The dataset is loaded using `librosa` for audio processing. Each audio file is in `.wav` format.
- **Feature extraction**: For each audio file, a **Mel-frequency spectrogram** is computed, which is then converted to decibel units using `librosa.power_to_db`.
- **Padding/truncation**: The spectrograms are padded or truncated to ensure they have a consistent size (44 time frames).
- **Normalization**: The spectrogram data is normalized using `StandardScaler` to scale the feature values to a standard range.
- **Label encoding**: Labels (words) are encoded into numerical values using `LabelEncoder` from `sklearn`.

### Data Split

The dataset is split into training and testing sets with an **80/20** ratio using `train_test_split` from `sklearn`.

---

## Instructions for Running the Code


1. **Download the dataset**:
    - You can download the **Google Speech Commands** dataset from the official website or use the Kaggle API if working on Kaggle.
    - Ensure the dataset is placed in the appropriate directory (specified by `dataset_dir` in the code).

2. **Install dependencies**:
    Install the required libraries using `pip`:
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the code**:
    Once the dataset is downloaded and dependencies are installed, run the script:
    ```bash
    python speech_command_recognition.py
    ```

    - The script will load the dataset, preprocess the audio files, train the CNN model, and evaluate its performance on the test set.

5. **Visualizations**:
    - The training and validation accuracy/loss curves will be displayed as plots.
    - A confusion matrix will also be displayed for evaluating the model's performance across different classes.

---

## Dependencies and Installation Instructions

### Python Version:
- Python 3.6 or higher.

### Required Libraries:
- `numpy`
- `librosa`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `keras`
- `tensorflow`

To install the required libraries, you can use the following command to install all dependencies:
```bash
pip install -r requirements.txt
