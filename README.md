# CCLM: The Contextual Caroline Language Model

## Introduction
This is a novel high-speed python language model architecture based upon Adamovsky's Caroline Word Graph (https://pages.pathcom.com/~vadco/cwg.html).
The base layer encodes words from a corpus based on their character composition.
A second layer stores relationships between words and captures their transition probabilties as weights.

## Usage
### Training a model
Prepare a text file corpus, one unlabeled sentence per line. Set:
```python
def train_test_dev_split(..., train_size=n)
```
where n is the desired total number of training sentences to randomly sample from the corpus file.

Set:
```python
ccwg = train_ccwg(train_sentences, window_size=10, freq_threshold=2, learning_rate=0.001, num_epochs=50)
```
to desired context window size, frequency threshold (words that appear < freq_threshold times in the training set are dropped from the model), learning rate for Adam, and number of epochs.

The script will prepare a test/train set from the corpus file and train a CCWG language model, then output two 50-word test sentences.

### Inference
TODO
