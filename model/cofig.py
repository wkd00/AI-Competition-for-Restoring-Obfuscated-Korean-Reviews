# config.py


class Config:
    seed = 42

    # 데이터 경로
    train_path = "../database/train/train.csv"
    test_path  = "../database/test/test.csv"

    # 학습 파라미터
    batch_size = 16
    num_epochs = 20
    lr = 1e-3

    # 모델 파라미터
    embed_dim = 512
    hidden_size = 512
    num_layers = 1  # GRU 레이어 수

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 특수 토큰
    PAD_TOKEN = "[PAD]"
    UNK_TOKEN = "[UNK]"
