# RNN Sentiment Analysis

This repository contains a Jupyter notebook for IMDb movie review sentiment classification with a simple recurrent neural network built in PyTorch.

## What it does

The notebook loads the IMDb dataset, cleans the review text, converts the labels to numeric form, vectorizes the reviews with TF-IDF, and trains an RNN to predict positive or negative sentiment.

## Workflow

1. Load the dataset from `IMDB Dataset.csv`.
2. Inspect the data, remove duplicates, and check for missing values.
3. Preprocess the review text by lowercasing, removing URLs, punctuation, HTML tags, and stop words, then applying stemming.
4. Encode sentiment labels and vectorize the cleaned text with TF-IDF.
5. Split the data into train and test sets.
6. Build and train a PyTorch RNN model.
7. Evaluate the model with accuracy on the test set.

## Files

- `RNN_sentiment_analysis.ipynb` - main notebook with the full workflow
- `IMDB Dataset.csv` - input dataset used by the notebook

## Requirements

The notebook uses:

- pandas
- numpy
- scikit-learn
- nltk
- torch

## How to run

1. Open `RNN_sentiment_analysis.ipynb` in Jupyter or VS Code.
2. Make sure `IMDB Dataset.csv` is in the same folder as the notebook.
3. Install the required Python packages if needed.
4. Run the notebook cells from top to bottom.

## Notes

- The notebook is written as an end-to-end experiment and is best run sequentially.
- NLTK stopwords and tokenizers may need to be downloaded before preprocessing.
