# model.py


class BiMultiLSTMModel(nn.Module):
    """
    양방향, 다층 LSTM 기반 모델
    """
    def __init__(self, vocab_size, embed_dim, hidden_size, num_layers, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        # 양방향 LSTM 적용
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True
        )

        # 양방향이므로 hidden_size * 2
        self.fc = nn.Linear(hidden_size * 2, vocab_size)

    def forward(self, x):
        x = self.embedding(x)  # (B, T, embed_dim)
        output, _ = self.lstm(x)  # (B, T, hidden_size * 2)
        logits = self.fc(output)  # (B, T, vocab_size)
        return logits

