import pandas as pd
import random
import hgtk  # 한글 자음/모음 분해 및 조합 라이브러리

# 데이터 로드
df_train = pd.read_csv("C:/Users/USER/OneDrive/Desktop/yonsei/daicon/hanguel_resolution/NKTG-Noised-Korean-Translator-by-GRU/database/train/train.csv")

# 확장된 한글 자음/모음 목록
CONSONANTS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
VOWELS = "ㅏㅑㅓㅕㅗㅛㅜㅠㅡㅣㅐㅔㅚㅟㅘㅙㅝㅞㅢ"
FINAL_CONSONANTS = "ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"

def add_typo_to_syllable(syllable, noise_rate=0.3):
    if not hgtk.checker.is_hangul(syllable): 
        return syllable 
    try: 
        decomposed = hgtk.letter.decompose(syllable) 
    except Exception: 
        return syllable

    if random.random() < noise_rate:
        modify_part = random.choice(["초성", "중성", "종성"])
        if modify_part == "초성" and decomposed[0] in CONSONANTS:
            candidates = [c for c in CONSONANTS if c != decomposed[0]]
            if candidates:
                new_cho = random.choice(candidates)
                if len(decomposed) == 3:
                    decomposed = (new_cho, decomposed[1], decomposed[2])
                else:
                    decomposed = (new_cho, decomposed[1])
        elif modify_part == "중성" and decomposed[1] in VOWELS:
            candidates = [v for v in VOWELS if v != decomposed[1]]
            if candidates:
                new_jung = random.choice(candidates)
                if len(decomposed) == 3:
                    decomposed = (decomposed[0], new_jung, decomposed[2])
                else:
                    decomposed = (decomposed[0], new_jung)
        elif modify_part == "종성":
            if len(decomposed) == 3 and decomposed[2]:
                candidates = [c for c in FINAL_CONSONANTS if c != decomposed[2]]
                if candidates:
                    new_jong = random.choice(candidates)
                    decomposed = (decomposed[0], decomposed[1], new_jong)
            else:
                if random.random() < 0.5:
                    new_jong = random.choice(FINAL_CONSONANTS)
                    decomposed = (decomposed[0], decomposed[1], new_jong)

    try:
        if len(decomposed) == 3 and decomposed[2]:
            return hgtk.letter.compose(*decomposed)
        else:
            return hgtk.letter.compose(decomposed[0], decomposed[1])
    except Exception:
        return syllable

def add_typo_to_word(word, noise_rate=0.3): # 단어의 각 음절마다 오타 적용 
    if all(char in (CONSONANTS + VOWELS) for char in word): 
        return word
    return "".join(add_typo_to_syllable(syllable, noise_rate) for syllable in word)

def add_typo_to_sentence(sentence, noise_rate=0.3): 
    words = sentence.split() 
    noisy_words = [add_typo_to_word(word, noise_rate) for word in words] 
    return " ".join(noisy_words)

def main(): # 데이터 로드 
    df_train = pd.read_csv("C:/Users/USER/OneDrive/Desktop/yonsei/daicon/hanguel_resolution/NKTG-Noised-Korean-Translator-by-GRU/database/train/train.csv")
    # 각 문장에 대해 오타 생성 (output 열을 바탕으로 input 열 생성)
    df_train["input"] = df_train["output"].apply(lambda x: add_typo_to_sentence(str(x), noise_rate=0.5))

    # 컬럼 정리 (ID, input, output 형식 유지)
    df_noisy = df_train[["ID", "input", "output"]]

    # 결과 저장
    output_path = "C:/Users/USER/OneDrive/Desktop/yonsei/daicon/hanguel_resolution/NKTG-Noised-Korean-Translator-by-GRU/database/train/noisy_train.csv"
    df_noisy.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"변환 완료, 저장 위치: {output_path}")


if __name__ == "__main__": 
    main()
