# dataset.py


class CharTokenizer:
    def __init__(self, vocab, pad_token, unk_token):
        self.vocab = vocab
        self.pad_token = pad_token
        self.unk_token = unk_token
        self.pad_id = vocab[pad_token]
        self.unk_id = vocab[unk_token]
        self.id2token = {v: k for k, v in vocab.items()}

    def encode(self, text):
        return [self.vocab.get(ch, self.unk_id) for ch in text]

    def decode(self, ids):
        return "".join([self.id2token.get(i, "") for i in ids if i in self.id2token])


class TokenClassifyDataset(Dataset):
    """
    난독화 해제용 (입출력 길이 무관) Dataset
    (input_str, output_str)을 (input_ids, label_ids)로 변환
    """
    def __init__(self, pairs, tokenizer):
        """
        pairs: list of (input_str, output_str)
        tokenizer: CharTokenizer
        """
        self.samples = []
        self.tokenizer = tokenizer

        for inp, outp in pairs:
            # 1) 전처리: 문장 끝의 공백 제거
            inp = inp.strip()
            outp = outp.strip()

            # 2) 토큰화
            input_ids = tokenizer.encode(inp)
            label_ids = tokenizer.encode(outp)

            # 3) 길이가 달라도 스킵하지 않고 그대로 저장
            self.samples.append((input_ids, label_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def build_vocab(pairs, special_tokens):
    """
    pairs: list of (input_str, output_str)
    special_tokens: [PAD, UNK] 등
    """
    chars = set()
    for inp, outp in pairs:
        chars.update(list(inp))
        chars.update(list(outp))

    vocab = {}
    for sp in special_tokens:
        vocab[sp] = len(vocab)
    for c in sorted(list(chars)):
        if c not in vocab:
            vocab[c] = len(vocab)
    return vocab


def token_collate_fn(batch):
    """
    batch: list of (input_ids, label_ids)
    """
    import torch
    input_list, label_list = [], []
    for inp, lab in batch:
        input_list.append(torch.tensor(inp, dtype=torch.long))
        label_list.append(torch.tensor(lab, dtype=torch.long))

    # pad_sequence => (B, T)
    input_padded = nn.utils.rnn.pad_sequence(
        input_list, batch_first=True, padding_value=0
    )
    label_padded = nn.utils.rnn.pad_sequence(
        label_list, batch_first=True, padding_value=0
    )
    return input_padded, label_padded
