"""
Contains the base Tokenizer class and a few common helper functions.
The base class also contains the (common) save/load functionality.
It would be possible to be a lot more strict about the interface and
e.g. isolating all regex/pattern parts to the RegexTokenizer, but
some concessions are made for simplicity.
"""
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import copy
import regex as re
import os
import pickle
from pretokenization_example import find_chunk_boundaries
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer
def iter_corpus(text: str, delimiter: str = r"<|endoftext|>"):
    delimiter = delimiter.replace('|', '\|')
    start = 0
    pattern = re.compile(rf'(.*?)\s*{delimiter}', re.DOTALL)
    for match in pattern.finditer(text):
        each_corpus = match.group(1).lstrip()
        if each_corpus:  # skip empty ones
            yield each_corpus
        start = match.end()
    # Yield trailing content if any
    remaining = text[start:].strip()
    if remaining:
        yield remaining

def get_pretoken_stats(corpus_iter):
    counts = defaultdict(int)
    for each_corpus in corpus_iter:
        #print('each_corpus', each_corpus)
        corpus_lst = re.finditer(GPT2_SPLIT_PATTERN, each_corpus)
        #print('corpus_lst', corpus_lst)
        for match in corpus_lst:
            word = match.group(0)
            word_bytes = word.encode("utf-8") 
            key = tuple(bytes([b]) for b in word_bytes)
            counts[key] += 1
    return counts
def process_chunk(args):
    filename, start, end, delimiter = args
    with open(filename, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        #print('chunk::::', chunk)
        corpus_iter = iter_corpus(chunk, delimiter)
        return get_pretoken_stats(corpus_iter)

def merge_dicts(dicts):
    #print('dicts', dicts)
    final = defaultdict(int)
    for d in dicts:
        for k, v in d.items():
            final[k] += v
    return final

def run_parallel_tokenization(file_path, num_processes=None, end_word= b"<|endoftext|>"):
    num_processes = num_processes or os.cpu_count()
    with open(file_path, 'rb') as f:
        boundaries = find_chunk_boundaries(f, num_processes, end_word)

    args_list = [
        (file_path, start, end, end_word.decode("utf-8"))
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        result_dicts = list(executor.map(process_chunk, args_list))

    final_counts = merge_dicts(result_dicts)
    return final_counts


def get_contigent_stats(words_freq):
    """
    Given a list of integers, return a dictionary of counts of consecutive pairs
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally allows to update an existing dictionary of counts
    """
    contigent_bytes_freq = defaultdict(int) 
    pair_position = defaultdict(set)
    
    for word_bytes, freq in words_freq.items():
        for i, pair in enumerate(zip(word_bytes, word_bytes[1:])): # iterate consecutive elements
            contigent_bytes_freq[pair] += 1 * freq
            #pair_position[pair].append((word_bytes, i, i+1))  # contigent_bytes_freq key, pair position start, pair position end
            pair_position[pair].add(word_bytes)

    # #pair_position_final {pair: {word_bytes: [(start_pos, end_pos), ...]}}
    # pair_position_final = dict()
    # for pair, lst in pair_position.items():
    #     pair_position_final[pair] = defaultdict(set)
    #     for word_bytes, start, end in lst:
    #         pair_position_final[pair][word_bytes].add(start)
        
    #pair_position (pair, {words})
    return contigent_bytes_freq, pair_position #pair_position_final


def merge(words_freq, pair, contigent_bytes_freq, pair_position):
    """
    change_words records the words relating to the selected pair
    {pair=(h, e): ((h, e, l, l o),(h,e,dd),(a,h,e,r))}
    Example: ids=[h, e, l, l o], pair=(h, e) -> [he, l, l, o]
    """
    change_words = pair_position[pair]
    new_pair_position = defaultdict(set, copy.deepcopy(pair_position))
    del new_pair_position[pair]
    # new_pair_position_final = copy.deepcopy(pair_position_final)
    # del new_pair_position_final[pair]
    print("change_words len::::", len(list(change_words)))
    for word_bytes in list(change_words):
        # refresh words_freq key based on change list
        i = 0
        new_word_bytes = []
        wfreq = words_freq[word_bytes]
        del words_freq[word_bytes]
        while i < len(word_bytes):

            print(word_bytes[i], i)
            if word_bytes[i]==pair[0] and i < len(word_bytes) - 1 and word_bytes[i+1] == pair[1]:
                new_word_bytes.append(word_bytes[i] + word_bytes[i+1])
                contigent_bytes_freq[pair] -= wfreq
                if contigent_bytes_freq[pair]==0:
                    del contigent_bytes_freq[pair]

                if i > 0:
                    #update contigent_bytes_freq for contigent position
                    contigent_bytes_freq[(word_bytes[i-1], word_bytes[i]+word_bytes[i+1])] +=wfreq
                    contigent_bytes_freq[(word_bytes[i-1], word_bytes[i])]-=wfreq
                    assert contigent_bytes_freq[(word_bytes[i-1], word_bytes[i])] >=0
                    if contigent_bytes_freq[(word_bytes[i-1], word_bytes[i])]==0:
                        del contigent_bytes_freq[(word_bytes[i-1], word_bytes[i])]
                    # add old word_bytes temparary, latter will replace with new_word_bytes when it is ready
                    new_pair_position[(word_bytes[i-1], word_bytes[i]+word_bytes[i+1])].add(word_bytes)
                    new_pair_position[(word_bytes[i-1], word_bytes[i])].discard(word_bytes)

                if i+1+1 < len(word_bytes) :
                    #update contigent_bytes_freq for contigent position
                    contigent_bytes_freq[(word_bytes[i]+word_bytes[i+1], word_bytes[i+2])] +=wfreq
                    contigent_bytes_freq[(word_bytes[i+1], word_bytes[i+2])]-=wfreq
                    assert contigent_bytes_freq[(word_bytes[i+1], word_bytes[i+2])] >= 0
                    if contigent_bytes_freq[(word_bytes[i+1], word_bytes[i+2])]==0:
                        del contigent_bytes_freq[(word_bytes[i+1], word_bytes[i+2])]
                    # add old word_bytes temparary, latter will replace with new_word_bytes when it is ready
                    print('pair show', (word_bytes[i]+word_bytes[i+1], word_bytes[i+2]))
                    new_pair_position[(word_bytes[i]+word_bytes[i+1], word_bytes[i+2])].add(word_bytes)
                    new_pair_position[(word_bytes[i+1], word_bytes[i+2])].discard(word_bytes)

                i+=2
            else:
                new_word_bytes.append(word_bytes[i])
                i+=1
        print('come out from first while loop')
        new_word_bytes = tuple(new_word_bytes)
        words_freq[new_word_bytes] = wfreq
        #print('new_pair_position before', new_pair_position)
        for each_pair in list(new_pair_position.keys()):
            if len(new_pair_position[each_pair])==0:
                del new_pair_position[each_pair]
                continue
            if word_bytes in new_pair_position[each_pair]:
                new_pair_position[each_pair].remove(word_bytes)
                new_pair_position[each_pair].add(new_word_bytes)

    return words_freq, contigent_bytes_freq, new_pair_position


# first two helper functions...
def replace_control_characters(s: str) -> str:
    # we don't want to print control characters
    # which distort the output (e.g. \n or much worse)
    # https://stackoverflow.com/questions/4324790/removing-control-characters-from-a-string-in-python/19016117#19016117
    # http://www.unicode.org/reports/tr44/#GC_Values_Table
    chars = []
    for ch in s:
        if unicodedata.category(ch)[0] != "C":
            chars.append(ch) # this character is ok
        else:
            chars.append(f"\\u{ord(ch):04x}") # escape
    return "".join(chars)

def render_token(t: bytes) -> str:
    # pretty print a token, escaping control characters
    s = t.decode('utf-8', errors='replace')
    s = replace_control_characters(s)
    return s

# -----------------------------------------------------------------------------
# the base Tokenizer class

class Tokenizer:
    """Base class for Tokenizers"""

    def __init__(self):
        # default: vocab size of 256 (all bytes), no merges, no patterns
        self.merges = [] # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = {} # int -> bytes
        self.reverse_vocab = {}

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    # def _build_vocab(self):
    #     # vocab is simply and deterministically derived from merges
    #     vocab = {idx: bytes([idx]) for idx in range(256)}
    #     for (p0, p1), idx in self.merges.items():
    #         vocab[idx] = vocab[p0] + vocab[p1]
    #     for special, idx in self.special_tokens.items():
    #         vocab[idx] = special.encode("utf-8")
    #     return vocab

    def save(self, model_file):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
     
        # Save multiple objects
        with open(model_file, "wb") as f:
            pickle.dump({
                "merges": self.merges,
                "vocab": self.vocab,
                "reverse_vocab": self.reverse_vocab,
                "special_tokens": self.special_tokens,
                "version": "1.0",
                "info": "Custom BPE model"
            }, f)
        print('---done---')
       

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        vocab = {}
        reverse_vocab = {}
        idx = 256
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        merges = model_data["merges"]
        vocab = model_data["vocab"]
        vocab = model_data["vocab"]
        reverse_vocab = model_data["reverse_vocab"]
        special_tokens = model_data["special_tokens"]
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = vocab
        self.reverse_vocab = reverse_vocab