import regex as re
from token_base import Tokenizer, run_parallel_tokenization, merge, get_contigent_stats, iter_corpus


# the main GPT text split patterns, see
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
          example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT2_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}
        #self.register_special_tokens({'<|endoftext|>': 100257})

    def train(self, path, vocab_size, num_processes=None, verbose=False, special_tokens={'<|endoftext|>': 100257}):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        self.register_special_tokens(special_tokens)

        # # input text preprocessing
        #print(list(self.special_tokens.keys)[0].encode("utf-8"))
        words_freq = run_parallel_tokenization(path, num_processes, list(self.special_tokens.keys())[0].encode("utf-8"))
        contigent_bytes_freq = None
        pair_position = None
        # iteratively merge the most common pairs to create new tokens
        #merges = {} # (int, int) -> int
        merges = []
        vocab = {idx: bytes([idx]) for idx in range(256)} # idx -> bytes
        reverse_vocab = {bytes([idx]):idx for idx in range(256)}
        for i in range(num_merges):
            # count the number of times every consecutive pair appears
            #stats = {}
            # for chunk_ids in ids:
            #     # passing in stats will update it in place, adding up counts
            #     get_stats(chunk_ids, stats)
            if not contigent_bytes_freq:
                contigent_bytes_freq, pair_position = get_contigent_stats(words_freq)
            # find the pair with the highest count
            pair = max(contigent_bytes_freq, key=lambda p: (contigent_bytes_freq[p], p))
            # for stats latter
            pair_counts = contigent_bytes_freq[pair]
            # mint a new token: assign it the next available id
            idx = 256 + i
            # replace all occurrences of pair in ids with idx
            #ids = [merge(chunk_ids, pair, idx) for chunk_ids in ids]
            words_freq, contigent_bytes_freq, pair_position = merge(words_freq, pair, contigent_bytes_freq, pair_position)
            # save the merge
            #merges[pair] = idx
            merges.append((pair[0] ,  pair[1]))
            vocab[idx] = pair[0] + pair[1]
            reverse_vocab[pair[0] + pair[1]] = idx
            # prints
            if verbose:
                print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {pair_counts} occurrences")

        # save class variables
        self.merges = merges 
        self.reverse_vocab = reverse_vocab # used in encode()
        self.vocab = vocab   # used in decode()
        
        return self.merges , self.vocab, self.reverse_vocab

    def register_special_tokens(self, special_tokens):
        # special_tokens is a dictionary of str -> int
        # example: {"<|endoftext|>": 100257}
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for k, v in special_tokens.items()}

    def encode(self, text_iter, merges, vocab, reverse_vocab, special_tokens={'<|endoftext|>': 100257}):
        if not self.special_tokens:
            self.register_special_tokens(special_tokens)
        spec_token = list(self.special_tokens.keys())[0]
        spec_token_idx = self.special_tokens[spec_token]
        for text in text_iter:
            corpus_iters = iter_corpus(text, spec_token.encode("utf-8"))
            for each_corpus in corpus_iters:
                #print('each_corpus', each_corpus)
                corpus_lst = re.finditer(GPT2_SPLIT_PATTERN, each_corpus)
                #print('corpus_lst', corpus_lst)
                for match in corpus_lst:
                    word = match.group(0)
                    word_bytes = word.encode("utf-8") 
                    word_bytes_lst = [bytes([b]) for b in word_bytes]
                    
                    for merge_pair in merges:
                        word_bytes_lst_temp = []
                        i=0
                        while i < len(word_bytes_lst) :
                            if word_bytes_lst[i]==merge_pair[0] and i+1<len(word_bytes_lst) and word_bytes_lst[i+1]==merge_pair[1] :#(word_bytes_lst[i], word_bytes_lst[i + 1]) == merge_pair:
                                merged = word_bytes_lst[i] + word_bytes_lst[i + 1]
                                word_bytes_lst_temp.append(merged)
                                i += 2
                                # Stay at i to check for new potential match just formed
                            
                            else:
                                word_bytes_lst_temp.append(word_bytes_lst[i])
                                i += 1
                        
                        word_bytes_lst = word_bytes_lst_temp
                    for b in word_bytes_lst:
                        yield self.reverse_vocab[b]
                    #word_encoded_lst = [self.reverse_vocab(b) for b in word_bytes_lst]
                yield spec_token_idx
    def decode(self, )
                


       
        



    def decode(self, ids):
        # given ids (list of integers), return Python string
        part_bytes = []
        for idx in ids:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode("utf-8"))
            else:
                raise ValueError(f"invalid token id: {idx}")
        text_bytes = b"".join(part_bytes)
        text = text_bytes.decode("utf-8", errors="replace")
        return text

    def _encode_chunk(self, text_bytes):
        # return the token ids
        # let's begin. first, convert all bytes to integers in range 0..255
        ids = list(text_bytes)
        while len(ids) >= 2:
            # find the pair with the lowest merge index
            stats = get_stats(ids)
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            # subtle: if there are no more merges available, the key will
            # result in an inf for every single pair, and the min will be
            # just the first pair in the list, arbitrarily
            # we can detect this terminating case by a membership check
            if pair not in self.merges:
                break # nothing else can be merged anymore
            # otherwise let's merge the best pair (lowest merge index)
            idx = self.merges[pair]
            ids = merge(ids, pair, idx)
        return ids


    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        # split text into chunks of text by categories defined in regex pattern
        text_chunks = re.findall(self.compiled_pattern, text)
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for chunk in text_chunks:
            chunk_bytes = chunk.encode("utf-8") # raw bytes
            chunk_ids = self._encode_chunk(chunk_bytes)
            ids.extend(chunk_ids)
        return ids

    def encode(self, text, allowed_special="none_raise"):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        # decode the user desire w.r.t. handling of special tokens
        special = None
        if allowed_special == "all":
            special = self.special_tokens
        elif allowed_special == "none":
            special = {}
        elif allowed_special == "none_raise":
            special = {}
            assert all(token not in text for token in self.special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f"allowed_special={allowed_special} not understood")
        if not special:
            # shortcut: if no special tokens, just use the ordinary encoding
            return self.encode_ordinary(text)
        # otherwise, we have to be careful with potential special tokens in text
        # we handle special tokens by splitting the text
        # based on the occurrence of any exact match with any of the special tokens
        # we can use re.split for this. note that surrounding the pattern with ()
        # makes it into a capturing group, so the special tokens will be included
        special_pattern = "(" + "|".join(re.escape(k) for k in special) + ")"
        special_chunks = re.split(special_pattern, text)
        # now all the special characters are separated from the rest of the text
        # all chunks of text are encoded separately, then results are joined
        ids = []
        for part in special_chunks:
            if part in special:
                # this is a special token, encode it separately as a special case
                ids.append(special[part])
            else:
                # this is an ordinary sequence, encode it normally
                ids.extend(self.encode_ordinary(part))
        return ids