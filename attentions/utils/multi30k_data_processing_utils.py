import re
import torch
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL_TOKENS = [PAD, BOS, EOS, UNK]
TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def word_tokenize(text):
    text = text.lower().strip()
    return TOKEN_RE.findall(text)


def build_vocab(tokenized_texts, max_vocab_size=10000, min_freq=3):
    counter = Counter(token for text in tokenized_texts for token in text)
    vocab = SPECIAL_TOKENS.copy()

    for token, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(vocab) >= max_vocab_size:
            break
        if token not in vocab:
            vocab.append(token)

    vocab_to_index = {token: index for index, token in enumerate(vocab)}
    index_to_vocab = {index: token for token, index in vocab_to_index.items()}

    return vocab_to_index, index_to_vocab


def preprocess(
    batch, source_lang, target_lang, source_vocab_to_index, target_vocab_to_index
):
    source_encodings = [
        encode(source_vocab_to_index, add_special_tokens(word_tokenize(text)))
        for text in batch[source_lang]
    ]
    target_encodings = [
        encode(target_vocab_to_index, add_special_tokens(word_tokenize(text)))
        for text in batch[target_lang]
    ]

    return {"source": source_encodings, "target": target_encodings}


def collate_fn(batch):
    source = [item["source"] for item in batch]
    target = [item["target"] for item in batch]

    source, source_lengths = pad_batch(source)  # defer padding till batching
    target, target_lengths = pad_batch(target)

    return {
        "source": source,
        "source_lengths": source_lengths,
        "target": target,
        "target_lengths": target_lengths,
    }


def add_special_tokens(tokens):
    return [BOS] + tokens + [EOS]


def remove_special_tokens(tokens):
    return [token for token in tokens if token not in [PAD, BOS, EOS]]


def encode(token_to_index, text):
    return [token_to_index.get(token, token_to_index[UNK]) for token in text]


def decode(index_to_token, indices):
    return " ".join(
        remove_special_tokens([index_to_token.get(index, UNK) for index in indices])
    )


def pad_batch(sequences, pad_idx=0):
    dtype = sequences[0].dtype

    lengths = torch.tensor([len(seq) for seq in sequences])
    max_length = int(lengths.max().item())

    padded_batch = torch.full((len(sequences), max_length), pad_idx, dtype=dtype)
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_batch[i, :end] = seq
    return padded_batch, lengths
