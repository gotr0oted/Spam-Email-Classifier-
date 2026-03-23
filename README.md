# Spam Email Classifier

This project builds a simple spam classifier using:

- `TF-IDF Vectorizer`
- `Multinomial Naive Bayes`
- a labeled `ham/spam` CSV dataset

## Files

- `spam_classifier.py` - trains the model and predicts a new message
- `spam_dataset.csv` - sample labeled dataset
- `requirements.txt` - Python dependencies

## Setup

```bash
pip install -r requirements.txt
```

## Run

Train the model:

```bash
python spam_classifier.py
```

Train and predict a new message:

```bash
python spam_classifier.py --message "Congratulations, you won a free prize"
```

## Expected Output

The script prints:

- model accuracy on a train/test split
- classification report
- prediction for the new message if `--message` is provided

