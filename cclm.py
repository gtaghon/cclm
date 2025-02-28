import numpy as np
import random
from collections import defaultdict
from os.path import isfile
from tqdm import tqdm
import shelve
from nltk.tokenize import word_tokenize

import matplotlib.pyplot as plt

class TrieNode:
    def __init__(self, char):
        self.char = char
        self.children = {}
        self.is_end_of_word = False
        self.pop_count = 0
        self.word_node = None

    def add_child(self, char):
        if char not in self.children:
            self.children[char] = TrieNode(char)
        return self.children[char]

class DAWGNode:
    def __init__(self, char):
        self.char = char
        self.children = {}
        self.is_end_of_word = False
        self.hash = None
        self.pop_count = 0
        self.word_node = None

    def add_child(self, char):
        if char not in self.children:
            self.children[char] = DAWGNode(char)
        return self.children[char]

class CWGNode:
    def __init__(self, char):
        self.char = char
        self.children = {}
        self.is_end_of_word = False
        self.hash = None
        self.pop_count = 0
        self.word_node = None

    def add_child(self, char):
        if char not in self.children:
            self.children[char] = CWGNode(char)
        return self.children[char]

class WordNode:
    def __init__(self, word):
        self.word = word
        self.contextual_edges = {}
        self.transition_probabilities = {}

    def add_contextual_edge(self, word_node, weight):
        self.contextual_edges[word_node] = weight

    def calculate_transition_probabilities(self):
        total_weight = sum(self.contextual_edges.values())
        self.transition_probabilities = {
            word_node: weight / total_weight
            for word_node, weight in self.contextual_edges.items()
        }

class CCWG:
    def __init__(self):
        self.root = TrieNode('')
        self.word_nodes = {}

    def add_word(self, word):
        node = self.root
        for char in word:
            node = node.add_child(char)
        node.is_end_of_word = True
        if word not in self.word_nodes:
            self.word_nodes[word] = WordNode(word)
        node.word_node = self.word_nodes[word]

    def compress_trie_to_dawg(self):
        print("Compressing Trie to DAWG...")
        
        queue = set()  # Use a set instead of a list
        self.root = self._compress_node(self.root, queue)
        self._assign_hash_values(self.root)
        
        print("Compression complete.")

    def _compress_node(self, node, queue):
        if node in queue:
            return queue[node]
        queue.add(node)

        new_node = DAWGNode(node.char)
        new_node.is_end_of_word = node.is_end_of_word
        new_node.pop_count = node.pop_count
        new_node.word_node = node.word_node

        for char, child_node in node.children.items():
            new_child_node = self._compress_node(child_node, queue)
            new_node.children[char] = new_child_node

        return new_node

    def _assign_hash_values(self, node):
        if node.is_end_of_word:
            node.hash = 1
        else:
            node.hash = 0

        for child_node in node.children.values():
            self._assign_hash_values(child_node)
            node.hash += child_node.hash

    def optimize_dawg_to_cwg(self):
        print("Optimizing DAWG to CWG...")
        self._optimize_node(self.root)
        print("Optimization complete.")

    def _optimize_node(self, node):
        new_node = CWGNode(node.char)
        new_node.is_end_of_word = node.is_end_of_word
        new_node.hash = node.hash
        new_node.word_node = node.word_node

        if len(node.children) > 0:
            sorted_children = sorted(node.children.items(), key=lambda x: x[0])
            child_hash_values = [child_node.hash for _, child_node in sorted_children]
            new_node.pop_count = self._calculate_pop_count(child_hash_values)
        else:
            new_node.pop_count = node.pop_count

        for char, child_node in node.children.items():
            new_child_node = self._optimize_node(child_node)
            new_node.children[char] = new_child_node

        return new_node

    def _calculate_pop_count(self, hash_values):
        def popcount(x):
            x = (x & 0x5555555555555555) + ((x >> 1) & 0x5555555555555555)
            x = (x & 0x3333333333333333) + ((x >> 2) & 0x3333333333333333)
            x = (x & 0x0F0F0F0F0F0F0F0F) + ((x >> 4) & 0x0F0F0F0F0F0F0F0F)
            x = (x & 0x00FF00FF00FF00FF) + ((x >> 8) & 0x00FF00FF00FF00FF)
            x = (x & 0x0000FFFF0000FFFF) + ((x >> 16) & 0x0000FFFF0000FFFF)
            x = (x & 0x00000000FFFFFFFF) + ((x >> 32) & 0x00000000FFFFFFFF)
            return x

        hash_mask = 0
        for hash_value in hash_values:
            hash_mask |= hash_value

        return popcount(hash_mask)

    def save_to_file(self, file_path):
        print("Saving CCWG to file...")
        
        if isfile(file_path):
            # If the file exists, load the existing database and update it
            with shelve.open(file_path, 'w') as db:
                existing_word_nodes = db.get('word_nodes', {})
                
                # Update the existing word nodes with the new ones
                for word, word_data in tqdm(self.word_nodes.items(), desc="Updating word nodes"):
                    if word not in existing_word_nodes:
                        existing_word_nodes[word] = {
                            'contextual_edges': {target_word.word: weight for target_word, weight in word_data.contextual_edges.items()},
                            'transition_probabilities': {target_word.word: prob for target_word, prob in word_data.transition_probabilities.items()}
                        }
                
                db['word_nodes'] = existing_word_nodes
        else:
            # If the file doesn't exist, create a new database
            with shelve.open(file_path, 'c') as db:
                # Save nodes
                db['nodes'] = self._serialize_nodes(self.root)
                
                # Save word nodes
                db['word_nodes'] = {word: {'contextual_edges': {target_word.word: weight for target_word, weight in word_node.contextual_edges.items()},
                                        'transition_probabilities': {target_word.word: prob for target_word, prob in word_node.transition_probabilities.items()}}
                                for word, word_node in tqdm(self.word_nodes.items(), desc="Saving word nodes")}
        
        print("CCWG saved to file.")

    def _serialize_nodes(self, node):
        serialized_node = {'char': node.char, 'is_end_of_word': node.is_end_of_word, 'hash': node.hash, 'pop_count': node.pop_count,
                        'children': {char: self._serialize_nodes(child_node) for char, child_node in node.children.items()}}
        return serialized_node

    @staticmethod
    def load_from_file(file_path):
        ccwg = CCWG()

        with shelve.open(file_path, 'r') as db:
            # Load nodes
            ccwg.root = ccwg._deserialize_nodes(db['nodes'])

            # Load word nodes
            ccwg.word_nodes = {}
            for word, word_data in db['word_nodes'].items():
                word_node = WordNode(word)
                ccwg.word_nodes[word] = word_node

            for word, word_data in db['word_nodes'].items():
                word_node = ccwg.word_nodes[word]
                word_node.contextual_edges = {ccwg.word_nodes[target_word]: weight
                                            for target_word, weight in word_data['contextual_edges'].items()}
                word_node.transition_probabilities = {ccwg.word_nodes[target_word]: prob
                                                    for target_word, prob in word_data['transition_probabilities'].items()}

        return ccwg

    def _deserialize_nodes(self, serialized_node):
        node = CWGNode(serialized_node['char'])
        node.is_end_of_word = serialized_node['is_end_of_word']
        node.hash = serialized_node['hash']
        node.pop_count = serialized_node['pop_count']
        node.children = {char: self._deserialize_nodes(child_node) for char, child_node in serialized_node['children'].items()}
        return node

    def tokenize_sentence(self, sentence):
        sentence = sentence.strip().lower()
        words = word_tokenize(sentence)
        return words

    def build_ccwg_from_sentences(self, sentences, batch_size=10000, window_size=5, freq_threshold=2):
            word_freq = defaultdict(int)
            
            batch = []
            for sentence in tqdm(sentences, desc="Building CCWG"):
                words = self.tokenize_sentence(sentence)
                
                for word in words:
                    word_freq[word] += 1
                
                batch.extend(words)
                
                if len(batch) >= batch_size:
                    self._process_batch(batch, window_size, freq_threshold, word_freq)
                    batch = []
            
            if batch:
                self._process_batch(batch, window_size, freq_threshold, word_freq)
            
            self.compress_trie_to_dawg()
            self.optimize_dawg_to_cwg()
            for word_node in self.word_nodes.values():
                word_node.calculate_transition_probabilities()

    def _process_batch(self, batch, window_size, freq_threshold, word_freq):
        for i, word in enumerate(batch):
            if word_freq[word] >= freq_threshold:
                self.add_word(word)
                for j in range(i - window_size, i + window_size + 1):
                    if j >= 0 and j < len(batch) and j != i:
                        context_word = batch[j]
                        if word_freq[context_word] >= freq_threshold:
                            if context_word not in self.word_nodes:
                                self.add_word(context_word)
                            self.word_nodes[word].add_contextual_edge(
                                self.word_nodes[context_word],
                                1
                            )

    def search_word(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_word

    def autocomplete(self, prefix):
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]
        return self._dfs_autocomplete(node, prefix)

    def _dfs_autocomplete(self, node, prefix):
        completions = []
        if node.is_end_of_word:
            completions.append(prefix)
        for char, child_node in node.children.items():
            completions.extend(self._dfs_autocomplete(child_node, prefix + char))
        return completions

    def generate_text(self, max_words=20, seed=None):
        if seed is not None:
            seed = seed.lower()
            if seed not in self.word_nodes:
                print(f"Seed word '{seed}' not found in the CCWG.")
                return ""
            current_word = self.word_nodes[seed]
            generated_words = [seed]
        else:
            current_word = random.choice(list(self.word_nodes.values()))
            generated_words = [current_word.word]

        for _ in range(max_words - 1):
            next_word = self._select_next_word(current_word)
            if next_word is None:
                break
            generated_words.append(next_word.word)
            current_word = next_word

        generated_text = " ".join(generated_words)
        if not generated_text:
            print("No text generated.")
        return generated_text

    def _select_next_word(self, word_node):
        if not word_node.transition_probabilities:
            return None
        next_word = random.choices(
            list(word_node.transition_probabilities.keys()),
            weights=list(word_node.transition_probabilities.values()),
            k=1
        )[0]
        return next_word
    
def generate_text_from_ccwg(model_file, max_words=50, seed=None):
    loaded_ccwg = CCWG.load_from_file(model_file)
    generated_text = loaded_ccwg.generate_text(max_words=max_words, seed=seed)
    print(generated_text)
    
class DataLoader:
    def __init__(self, sentences, batch_size):
        self.sentences = sentences
        self.batch_size = batch_size

    def __iter__(self):
        random.shuffle(self.sentences)
        for i in range(0, len(self.sentences), self.batch_size):
            batch = self.sentences[i:i+self.batch_size]
            yield batch

def train_test_dev_split(corpus_file, train_size=55000, test_ratio=0.1, dev_ratio=0.1):
    with open(corpus_file, 'r') as file:
        sentences = file.readlines()
    random.shuffle(sentences)
    
    train_sentences = sentences[:train_size]
    test_size = int(len(train_sentences) * test_ratio)
    dev_size = int(len(train_sentences) * dev_ratio)
    
    test_sentences = train_sentences[:test_size]
    dev_sentences = train_sentences[test_size:test_size+dev_size]
    train_sentences = train_sentences[test_size+dev_size:]
    
    return train_sentences, test_sentences, dev_sentences

def calculate_perplexity(model_file, test_sentences):
    ### note Saving disabled here
    loaded_ccwg = ccwg #CCWG.load_from_file(model_file)
    total_log_prob = 0
    total_words = 0
    for sentence in test_sentences:
        words = loaded_ccwg.tokenize_sentence(sentence)
        sentence_log_prob = 0
        for i in range(1, len(words)):
            current_word = loaded_ccwg.word_nodes.get(words[i])
            prev_word = loaded_ccwg.word_nodes.get(words[i-1])
            if current_word and prev_word:
                transition_prob = prev_word.transition_probabilities.get(current_word, 0)
                if transition_prob > 0:
                    sentence_log_prob += np.log(transition_prob)
                else:
                    sentence_log_prob += np.log(1e-8)  # Smoothing for unseen transitions
            else:
                sentence_log_prob += np.log(1e-8)  # Smoothing for unknown words
        total_log_prob += sentence_log_prob
        total_words += len(words)
    perplexity = np.exp(-total_log_prob / total_words)
    return perplexity

def optimize_transition_probabilities(ccwg, learning_rate=0.001, num_epochs=10):
    for epoch in tqdm(range(num_epochs), desc = 'Training with Adam'):
        for word_node in ccwg.word_nodes.values():
            if word_node.transition_probabilities:
                m = 0
                v = 0
                beta1 = 0.9
                beta2 = 0.999
                epsilon = 1e-8
                
                for target_word, prob in word_node.transition_probabilities.items():
                    g = -np.log(prob)
                    m = beta1 * m + (1 - beta1) * g
                    v = beta2 * v + (1 - beta2) * g**2
                    m_hat = m / (1 - beta1**(epoch+1))
                    v_hat = v / (1 - beta2**(epoch+1))
                    word_node.transition_probabilities[target_word] = np.exp(-learning_rate * m_hat / (np.sqrt(v_hat) + epsilon))

def train_ccwg(train_sentences, window_size=5, freq_threshold=2, learning_rate=0.001, num_epochs=10):
    ccwg = CCWG()
    ccwg.build_ccwg_from_sentences(train_sentences, window_size=window_size, freq_threshold=freq_threshold)
    
    optimize_transition_probabilities(ccwg, learning_rate=learning_rate, num_epochs=num_epochs)
    
    ### Uncomment this line to save model files to repo root directory
    # ccwg.save_to_file('ccwg_model')
    
    return ccwg

if __name__ == '__main__':
    
    contexts = [4, 8, 16, 32, 64, 128, 256]
    perplexities = []
    
    for context in contexts:
        corpus_file = 'dracula.txt'
        model_file = 'ccwg_model_dracula.db'

        train_sentences, test_sentences, dev_sentences = train_test_dev_split(corpus_file)

        # Train the CCWG model
        ccwg = train_ccwg(train_sentences, window_size=context, freq_threshold=2, learning_rate=0.001, num_epochs=10)

        # Calculate perplexity on the test set
        perplexity = calculate_perplexity(model_file, test_sentences)
        print(f"Test Perplexity: {perplexity:.2f}")

            # Generate text using the trained model
        print(ccwg.generate_text(max_words=10, seed='the'))
        print(ccwg.generate_text(max_words=10, seed='another'))
        
        # DF appends
        perplexities.append(perplexity)
        
    plt.plot(contexts, perplexities)
    plt.xlabel("context window size")
    plt.ylabel("perplexity")
    plt.title("CCWG on Dracula")
    plt.show()
