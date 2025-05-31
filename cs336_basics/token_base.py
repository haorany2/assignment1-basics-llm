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
from pretokenization_example import find_chunk_boundaries
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

# -----------------------------------------------------------------------------
# a few helper functions useful for both BasicTokenizer and RegexTokenizer
def iter_corpus(text: str, delimiter: str = r"<\|endoftext\|>"):
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
    filename, start, end = args
    with open(filename, 'rb') as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        #print('chunk::::', chunk)
        corpus_iter = iter_corpus(chunk)
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
        (file_path, start, end)
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
    In the list of integers (ids), replace all consecutive occurrences
    of pair with the new integer token idx
    Example: ids=[h, e, l, l o], pair=(h, e) -> [he, l, l, o]
    """
    change_words = pair_position[pair]
    new_pair_position = copy.deepcopy(pair_position)
    del new_pair_position[pair]
    # new_pair_position_final = copy.deepcopy(pair_position_final)
    # del new_pair_position_final[pair]

    for word_bytes in list(change_words):
        # refresh words_freq key based on change list
        i = 0
        new_word_bytes = []
        wfreq = words_freq[word_bytes]
        del words_freq[word_bytes]
        while i < len(word_bytes):
            if word_bytes[i]==pair[0] and i < len(word_bytes) - 1 and word_bytes[i+1] == pair[1]:
                new_word_bytes.append(word_bytes[i]+word_bytes[i+1])
                contigent_bytes_freq[pair] -= wfreq
                if contigent_bytes_freq[pair]==0:
                    del contigent_bytes_freq[pair]

                if i > 0:
                    #update contigent_bytes_freq for contigent position
                    contigent_bytes_freq[(word_bytes[i-1], word_bytes[i]+word_bytes[i+1])] +=wfreq
                    contigent_bytes_freq[(word_bytes[i-1], word_bytes[i])]-=wfreq
                    if contigent_bytes_freq[(word_bytes[i-1], word_bytes[i])]==0:
                        del contigent_bytes_freq[(word_bytes[i-1], word_bytes[i])]
                    # add old word_bytes temparary, latter will replace with new_word_bytes when it is ready
                    new_pair_position[(word_bytes[i-1], word_bytes[i]+word_bytes[i+1])].add(word_bytes)
                    new_pair_position[(word_bytes[i-1], word_bytes[i])].remove(word_bytes)

                if i+1+1 < len(word_bytes) :
                    #update contigent_bytes_freq for contigent position
                    contigent_bytes_freq[(word_bytes[i]+word_bytes[i+1], word_bytes[i+2])] +=wfreq
                    contigent_bytes_freq[(word_bytes[i+1], word_bytes[i+2])]-=wfreq
                    if contigent_bytes_freq[(word_bytes[i+1], word_bytes[i+2])]==0:
                        del contigent_bytes_freq[(word_bytes[i+1], word_bytes[i+2])]
                    # add old word_bytes temparary, latter will replace with new_word_bytes when it is ready
                    new_pair_position[(word_bytes[i]+word_bytes[i+1], word_bytes[i+2])].add(word_bytes)
                    new_pair_position[(word_bytes[i+1], word_bytes[i+2])].remove(word_bytes)

                i+=2
            else:
                new_word_bytes.append(word_bytes[i])
                i+=1
        new_word_bytes = tuple(new_word_bytes)
        words_freq[new_word_bytes] = wfreq
        print('new_pair_position before', new_pair_position)
        for each_pair in list(new_pair_position.keys()):
            if len(new_pair_position[each_pair])==0:
                del new_pair_position[each_pair]
                continue
            if word_bytes in new_pair_position[each_pair]:
                new_pair_position[each_pair].remove(word_bytes)
                new_pair_position[each_pair].add(new_word_bytes)

       
       
        # new_word_bytes = list(word_bytes)
        # new_word_bytes[start] = pair[0] + pair[1]
        # del new_word_bytes[end]
        # new_word_bytes = tuple(new_word_bytes)
        # refresh words_freq 
        
        
       
        

        # # # refresh new_pair_position_final
        # # new_pair_position_final[new_word_bytes] = defaultdict(list)
        
        # for start in list(start_pos_set):
            

        #     # refresh contigent_bytes_freq 
        #     contigent_bytes_freq[pair] -= wfreq
        #     if start-1>=0:
        #         contigent_bytes_freq[(new_word_bytes[start-1],new_word_bytes[start])] += wfreq
                
        #         # # refresh new_pair_position_final
        #         # if new_word_bytes[start-1]+new_word_bytes[start] not in new_pair_position_final:
        #         #     new_pair_position_final[new_word_bytes[start-1]+new_word_bytes[start]] = dict()
        #         # new_pair_position_final[new_word_bytes[start-1]+new_word_bytes[start]][new_word_bytes].append((new_word_bytes[start-1],new_word_bytes[start]))
        #         # past + pair  pair+next
        #     if start+1<len(new_word_bytes):
        #         contigent_bytes_freq[(new_word_bytes[start], new_word_bytes[start+1])] += wfreq
                
        #         # # refresh new_pair_position_final
        #         # new_pair_position_final[new_word_bytes].append((new_word_bytes[start], new_word_bytes[start+1]))
        
        # # # refresh new_pair_position_final
        # # for each_pair, pos_dict in new_pair_position_final.items():
        # #     if word_bytes in pos_dict.keys():
        # #         del new_pair_position_final[each_pair][word_bytes]
        # #     if new_word_bytes not in new_pair_position_final[each_pair]:
        # #         new_pair_position_final[each_pair][new_word_bytes] = defaultdict(set)
        # # for i, new_pair in enumerate(zip(new_word_bytes, new_word_bytes[1:])):
        # #     new_pair_position_final[new_pair][new_word_bytes].append((i,i+1))


    # if contigent_bytes_freq[pair]==0:
    #     del contigent_bytes_freq[pair]
    return words_freq, contigent_bytes_freq, new_pair_position


    # i = 0
    # while i < len(word_bytes):
    #     # if not at the very last position AND the pair matches, replace it
    #     if word_bytes[i] == pair[0] and i < len(word_bytes) - 1 and word_bytes[i+1] == pair[1]:
    #         new_word_bytes.append(pair[0]+pair[1])
    #         i += 2
    #     else:
    #         new_word_bytes.append(word_bytes[i])
    #         i += 1
    # return tuple(new_word_bytes)

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
        self.merges = {} # (int, int) -> int
        self.pattern = "" # str
        self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
        self.vocab = self._build_vocab() # int -> bytes

    def train(self, text, vocab_size, verbose=False):
        # Tokenizer can train a vocabulary of size vocab_size from text
        raise NotImplementedError

    def encode(self, text):
        # Tokenizer can encode a string into a list of integers
        raise NotImplementedError

    def decode(self, ids):
        # Tokenizer can decode a list of integers into a string
        raise NotImplementedError

    def _build_vocab(self):
        # vocab is simply and deterministically derived from merges
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for (p0, p1), idx in self.merges.items():
            vocab[idx] = vocab[p0] + vocab[p1]
        for special, idx in self.special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab

    def save(self, file_prefix):
        """
        Saves two files: file_prefix.vocab and file_prefix.model
        This is inspired (but not equivalent to!) sentencepiece's model saving:
        - model file is the critical one, intended for load()
        - vocab file is just a pretty printed version for human inspection only
        """
        # write the model: to be used in load() later
        model_file = file_prefix + ".model"
        with open(model_file, 'w') as f:
            # write the version, pattern and merges, that's all that's needed
            f.write("minbpe v1\n")
            f.write(f"{self.pattern}\n")
            # write the special tokens, first the number of them, then each one
            f.write(f"{len(self.special_tokens)}\n")
            for special, idx in self.special_tokens.items():
                f.write(f"{special} {idx}\n")
            # the merges dict
            for idx1, idx2 in self.merges:
                f.write(f"{idx1} {idx2}\n")
        # write the vocab: for the human to look at
        vocab_file = file_prefix + ".vocab"
        inverted_merges = {idx: pair for pair, idx in self.merges.items()}
        with open(vocab_file, "w", encoding="utf-8") as f:
            for idx, token in self.vocab.items():
                # note: many tokens may be partial utf-8 sequences
                # and cannot be decoded into valid strings. Here we're using
                # errors='replace' to replace them with the replacement char �.
                # this also means that we couldn't possibly use .vocab in load()
                # because decoding in this way is a lossy operation!
                s = render_token(token)
                # find the children of this token, if any
                if idx in inverted_merges:
                    # if this token has children, render it nicely as a merge
                    idx0, idx1 = inverted_merges[idx]
                    s0 = render_token(self.vocab[idx0])
                    s1 = render_token(self.vocab[idx1])
                    f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
                else:
                    # otherwise this is leaf token, just print it
                    # (this should just be the first 256 tokens, the bytes)
                    f.write(f"[{s}] {idx}\n")

    def load(self, model_file):
        """Inverse of save() but only for the model file"""
        assert model_file.endswith(".model")
        # read the model file
        merges = {}
        special_tokens = {}
        idx = 256
        with open(model_file, 'r', encoding="utf-8") as f:
            # read the version
            version = f.readline().strip()
            assert version == "minbpe v1"
            # read the pattern
            self.pattern = f.readline().strip()
            # read the special tokens
            num_special = int(f.readline().strip())
            for _ in range(num_special):
                special, special_idx = f.readline().strip().split()
                special_tokens[special] = int(special_idx)
            # read the merges
            for line in f:
                idx1, idx2 = map(int, line.split())
                merges[(idx1, idx2)] = idx
                idx += 1
        self.merges = merges
        self.special_tokens = special_tokens
        self.vocab = self._build_vocab()