# CCLM: The Contextual Caroline Language Model

## Introduction
This is a novel high-speed python language model architecture based upon Adamovsky's Caroline Word Graph (https://pages.pathcom.com/~vadco/cwg.html).
The base layer encodes words from a corpus based on their character composition.
A second layer stores relationships between words and captures their transition probabilties as weights.

Two corpora are included for testing, a portion of the NLTK Brown Corpus and the fulltext of Bram Stoker's Dracula from Project Gutenberg.

## Usage
### Training a model
Prepare a text file corpus, one unlabeled sentence per line. Set:
```python
def train_test_dev_split(..., train_size=n)
```
where n is the desired total number of training sentences to randomly sample from the corpus file. The default train/dev/test split is 80/10/10.

The main script as-is will prepare a test/train set from the corpus file and train CCWG language models of increasing context window size (4-256 words), outputting the average perplexity across the test set and two 10-word test sentences starting from single words. When complete, you'll see a plot of perplexity vs. context window size for your test set.

### Customization
If you want to script training for yourself, set:
```python
ccwg = train_ccwg(train_sentences, window_size=8, freq_threshold=2, learning_rate=0.001, num_epochs=10)
```
to desired context window size, frequency threshold (words that appear < freq_threshold times in the training set are dropped from the model), learning rate for Adam, and number of epochs.