import random
import pandas as pd
import re

def apply_liaison(text: str) -> str:
    """
    한글 문자열에 대해 연음 규칙(받침이 있는 음절의 받침이 뒤 음절의 초성 'ㅇ'으로
    넘어가는 현상)을 적용하여 재구성합니다.

    예시:
      입력: "밥을 먹어"  ->  출력(발음표기 예시): "바블 먹어"

    주의: 이 코드는 기본적인 단일 받침 및 일부 복합받침(ㄳ, ㄵ 등)에 대해
          첫번째 자모만 옮기는 방식으로 처리합니다.
    """
    BASE_CODE = 0xAC00  # '가'의 유니코드
    # 한글 음절 구성: 초성 19개, 중성 21개, 종성 28개
    NUM_JUNG = 21
    NUM_JONG = 28

    # 각 음절을 (초성, 중성, 종성)으로 분해한 정보를 담을 리스트
    syllables = []
    for char in text:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:  # 한글 음절 범위
            syl_index = code - BASE_CODE
            cho = syl_index // (NUM_JUNG * NUM_JONG)
            jung = (syl_index % (NUM_JUNG * NUM_JONG)) // NUM_JONG
            jong = syl_index % NUM_JONG
            syllables.append({"is_hangul": True, "cho": cho, "jung": jung, "jong": jong})
        else:
            syllables.append({"is_hangul": False, "char": char})

    # 받침(종성)에서 초성으로의 변환 매핑 (단일 받침)
    jong_to_cho = {
        1: 0,   # ㄱ
        2: 1,   # ㄲ
        4: 2,   # ㄴ
        7: 3,   # ㄷ
        8: 5,   # ㄹ
        16: 6,  # ㅁ
        17: 7,  # ㅂ
        19: 9,  # ㅅ
        20: 10, # ㅆ
        21: 11, # ㅇ
        22: 12, # ㅈ
        23: 14, # ㅊ
        24: 15, # ㅋ
        25: 16, # ㅌ
        26: 17, # ㅍ
        27: 18  # ㅎ
    }
    # 복합 받침일 경우, 첫 자모를 기준으로 함 (예: ㄳ → ㄱ)
    compound_mapping = {
        3: 0,   # ㄳ -> ㄱ
        5: 2,   # ㄵ -> ㄴ
        6: 2,   # ㄶ -> ㄴ
        9: 5,   # ㄺ -> ㄹ
        10: 5,  # ㄻ -> ㄹ
        11: 5,  # ㄼ -> ㄹ
        12: 5,  # ㄽ -> ㄹ
        13: 5,  # ㄾ -> ㄹ
        14: 5,  # ㄿ -> ㄹ
        15: 5,  # ㅀ -> ㄹ
        18: 7   # ㅄ -> ㅂ
    }

    # 인접한 음절에 대해 연음 규칙 적용
    # 조건: 앞 음절에 종성이 있고, 뒤 음절의 초성이 'ㅇ'(인덱스 11)일 경우
    for i in range(len(syllables) - 1):
        current = syllables[i]
        nxt = syllables[i+1]
        if current["is_hangul"] and nxt["is_hangul"]:
            if current["jong"] != 0 and nxt["cho"] == 11:
                jong_val = current["jong"]
                # 단일 받침인지, 복합 받침인지를 확인하여 초성 인덱스로 변환
                if jong_val in jong_to_cho:
                    new_cho = jong_to_cho[jong_val]
                elif jong_val in compound_mapping:
                    new_cho = compound_mapping[jong_val]
                else:
                    continue  # 변환할 수 없는 경우 넘어감
                # 연음 규칙 적용: 앞 음절의 받침은 사라지고, 뒤 음절의 초성이 변환된 받침이 됨
                current["jong"] = 0
                nxt["cho"] = new_cho

    # 음절 정보를 다시 한글 음절로 조합
    result_chars = []
    for syl in syllables:
        if syl["is_hangul"]:
            new_code = BASE_CODE + (syl["cho"] * NUM_JUNG + syl["jung"]) * NUM_JONG + syl["jong"]
            result_chars.append(chr(new_code))
        else:
            result_chars.append(syl["char"])

    return "".join(result_chars)

def change_initials(text: str) -> str:
    """
    입력된 문장의 각 한글 음절의 초성에 대해 다음 규칙을 적용:
      - 초성이 ㄴ(인덱스 2) 또는 ㅁ(인덱스 6)이면 그대로 둠.
      - 그 외의 초성은 50% 확률로 변경. (변경할 경우, 아래 그룹 내에서 다른 자모로 무조건 바꿈)
         그룹은 다음과 같다.
           그룹1: ㄷ(3), ㄸ(4), ㅌ(16)
           그룹2: ㅈ(12), ㅉ(13), ㅊ(14)
           그룹3: ㄱ(0), ㄲ(1), ㅋ(15)
           그룹4: ㅂ(7), ㅃ(8), ㅍ(17)
           그룹5: ㅅ(9), ㅆ(10)
           그룹6: ㅇ(11), ㅎ(18)

      ※ 단, 초성이 위 그룹에 속하지 않는 경우(예: ㄹ, 인덱스 5)는 변경하지 않음.
    """
    BASE_CODE = 0xAC00  # '가'
    NUM_JUNG = 21
    NUM_JONG = 28

    # 한글 음절의 초성은 아래 순서(인덱스)로 구성됨.
    # [ㄱ(0), ㄲ(1), ㄴ(2), ㄷ(3), ㄸ(4), ㄹ(5), ㅁ(6), ㅂ(7), ㅃ(8),
    #  ㅅ(9), ㅆ(10), ㅇ(11), ㅈ(12), ㅉ(13), ㅊ(14), ㅋ(15), ㅌ(16), ㅍ(17), ㅎ(18)]

    # 정의된 그룹 (각 그룹은 초성 인덱스 리스트)
    groups = [
        [3, 4, 16],    # ㄷ, ㄸ, ㅌ
        [12, 13, 14],  # ㅈ, ㅉ, ㅊ
        [0, 1, 15],    # ㄱ, ㄲ, ㅋ
        [7, 8, 17],    # ㅂ, ㅃ, ㅍ
        [9, 10],       # ㅅ, ㅆ
        [11, 18]       # ㅇ, ㅎ
    ]
    # 초성이 속한 그룹을 빠르게 찾기 위한 매핑 생성
    group_map = {}
    for grp in groups:
        for init in grp:
            group_map[init] = grp

    result_chars = []

    for char in text:
        code = ord(char)
        # 한글 음절인지 확인 (가 ~ 힣)
        if 0xAC00 <= code <= 0xD7A3:
            syllable_index = code - BASE_CODE
            cho = syllable_index // (NUM_JUNG * NUM_JONG)
            jung = (syllable_index % (NUM_JUNG * NUM_JONG)) // NUM_JONG
            jong = syllable_index % NUM_JONG

            # ㄴ(2)와 ㅁ(6)은 그대로 둠
            if cho in (2, 6):
                pass
            else:
                # 50% 확률로 변경 여부 결정
                if random.random() >= 0.5:
                    # 변경할 초성이 그룹에 속하는지 확인
                    if cho in group_map:
                        # 같은 그룹 내에서 현재 초성을 제외한 다른 자모로 무조건 변경
                        alternatives = [init for init in group_map[cho] if init != cho]
                        if alternatives:  # alternatives가 있을 때
                            cho = random.choice(alternatives)
                    # 만약 초성이 그룹에 속하지 않으면 (예: ㄹ, 인덱스 5) 변경하지 않음.

            # 새 음절 재구성
            new_code = BASE_CODE + (cho * NUM_JUNG + jung) * NUM_JONG + jong
            result_chars.append(chr(new_code))
        else:
            # 한글 음절이 아니면 그대로 추가
            result_chars.append(char)

    return "".join(result_chars)

# 한글 음절의 구성: 초성 19, 중성 21, 종성 28
BASE_CODE = 0xAC00  # '가'
NUM_JUNG = 21
NUM_JONG = 28

def change_medial(jung: int) -> int:
    """
    주어진 중성(jung, 0~20)에 대해 아래의 확률 규칙을 적용하여
    변환한 중성 인덱스를 반환합니다.
    """
    # 그룹: ㅐ, ㅔ, ㅒ, ㅖ, ㅙ, ㅞ, ㅚ (인덱스: 1, 3, 5, 7, 10, 11, 15)
    group_6 = [1, 3, 5, 7, 10, 11, 15]
    if jung in group_6:
        # 50% 확률로 그룹 내 다른 모음으로 변경
        if random.random() < 0.5:
            alternatives = [v for v in group_6 if v != jung]
            return random.choice(alternatives)
        else:
            return jung
    elif jung == 4:  # ㅓ
        if random.random() < 0.5:
            return random.choice([6, 14])  # ㅕ 또는 ㅝ
        else:
            return jung
    elif jung == 6:  # ㅕ
        if random.random() < 0.5:
            return 4  # ㅓ
        else:
            return jung
    elif jung == 14:  # ㅝ
        if random.random() < 0.25:
            return 4  # ㅓ
        else:
            return jung
    elif jung == 18:  # ㅡ
        if random.random() < 0.5:
            return random.choice([19, 13])  # ㅢ 또는 ㅜ
        else:
            return jung
    elif jung == 19:  # ㅢ
        if random.random() < 0.5:
            return 18  # ㅡ
        else:
            return jung
    elif jung == 13:  # ㅜ
        if random.random() < 0.5:
            return random.choice([18, 17])  # ㅡ 또는 ㅠ
        else:
            return jung
    elif jung == 17:  # ㅠ
        if random.random() < 0.5:
            return 13  # ㅜ
        else:
            return jung
    elif jung == 0:  # ㅏ
        if random.random() < 0.5:
            return random.choice([2, 9])  # ㅑ 또는 ㅘ
        else:
            return jung
    elif jung == 2:  # ㅑ
        if random.random() < 0.5:
            return 0  # ㅏ
        else:
            return jung
    elif jung == 9:  # ㅘ
        if random.random() < 0.5:
            return 0  # ㅏ
        else:
            return jung
    elif jung == 20:  # ㅣ
        if random.random() < 0.5:
            return random.choice([16, 19])  # ㅟ 또는 ㅢ
        else:
            return jung
    elif jung == 16:  # ㅟ
        if random.random() < 0.25:
            return 20  # ㅣ
        else:
            return jung
    else:
        return jung

def change_medials(text: str) -> str:
    """
    입력된 문장의 각 한글 음절의 중성을 위의 규칙에 따라 변경합니다.
    한글 음절은 유니코드 0xAC00 ~ 0xD7A3 범위에 있습니다.
    """
    result_chars = []
    for char in text:
        code = ord(char)
        if 0xAC00 <= code <= 0xD7A3:
            syl_index = code - BASE_CODE
            cho = syl_index // (NUM_JUNG * NUM_JONG)
            jung = (syl_index % (NUM_JUNG * NUM_JONG)) // NUM_JONG
            jong = syl_index % NUM_JONG

            new_jung = change_medial(jung)
            new_code = BASE_CODE + (cho * NUM_JUNG + new_jung) * NUM_JONG + jong
            result_chars.append(chr(new_code))
        else:
            result_chars.append(char)
    return "".join(result_chars)

# 한글 음절 구성 상수
BASE_CODE = 0xAC00  # '가'
NUM_JUNG = 21
NUM_JONG = 28

# 종성이 없는 경우 채워넣을 수 있는 종성 (인덱스 1 ~ 27)
possible_jongs = list(range(1, 28))

# 종성 변경 규칙 사전 (현재 종성 인덱스: 대체 가능한 종성 인덱스 리스트)
jong_transformation = {
    1: [2, 3, 24],       # ㄱ -> ㄲ, ㄳ, ㅋ
    2: [1, 3, 24],       # ㄲ -> ㄱ, ㄳ, ㅋ
    3: [1, 2, 24],       # ㄳ -> ㄱ, ㄲ, ㅋ
    4: [5, 6],          # ㄴ -> ㄵ, ㄶ
    5: [4],             # ㄵ -> ㄴ
    6: [4],             # ㄶ -> ㄴ
    7: [25],            # ㄷ -> ㅌ
    8: [9, 10, 11, 12, 13, 14, 15],  # ㄹ -> ㄺ, ㄻ, ㄼ, ㄽ, ㄾ, ㄿ, ㅀ
    9: [8, 1],          # ㄺ -> ㄹ, ㄱ
    10: [8, 16],        # ㄻ -> ㄹ, ㅁ
    11: [8, 17],        # ㄼ -> ㄹ, ㅂ
    12: [8, 19],        # ㄽ -> ㄹ, ㅅ
    13: [8, 25],        # ㄾ -> ㄹ, ㅌ
    14: [8, 26],        # ㄿ -> ㄹ, ㅍ
    15: [8, 27],        # ㅀ -> ㄹ, ㅎ
    # 16 (ㅁ)는 특별히 그대로 유지
    17: [11, 18, 26],    # ㅂ -> ㄼ, ㅄ, ㅍ
    18: [17, 26],        # ㅄ -> ㅂ, ㅍ
    19: [3, 12, 18, 20],  # ㅅ -> ㄳ, ㄽ, ㅄ, ㅆ
    20: [19],            # ㅆ -> ㅅ
    21: [27],            # ㅇ -> ㅎ
    22: [5, 23],         # ㅈ -> ㄵ, ㅊ
    23: [22],            # ㅊ -> ㅈ
    24: [1, 2],          # ㅋ -> ㄱ, ㄲ
    25: [7, 13],         # ㅌ -> ㄷ, ㄾ
    26: [17, 14],        # ㅍ -> ㅂ, ㄿ
    27: [15, 21]         # ㅎ -> ㅀ, ㅇ
}

def change_final(text: str) -> str:
    """
    주어진 문자열의 각 한글 음절에 대해 종성을 아래 규칙에 따라 변경합니다.

    1. 종성이 없는 경우:
       50% 확률로 가능한 종성(인덱스 1~27) 중 무작위로 채워넣습니다.

    2. 종성이 있는 경우:
       - 만약 종성이 ㅁ(인덱스 16)라면 그대로 유지합니다.
       - 그 외의 경우 33% 확률로 원래 종성을 유지, 33% 확률로 종성을 삭제(0),
         그리고 나머지 33% 확률로 아래 규칙에 따라 종성을 변경합니다.
    """
    result_chars = []

    for char in text:
        code = ord(char)
        # 한글 음절(가 ~ 힣)인지 확인
        if 0xAC00 <= code <= 0xD7A3:
            syl_index = code - BASE_CODE
            cho = syl_index // (NUM_JUNG * NUM_JONG)
            jung = (syl_index % (NUM_JUNG * NUM_JONG)) // NUM_JONG
            jong = syl_index % NUM_JONG

            if jong == 0:
                # 종성이 없는 경우: 30% 확률로 무작위 종성 추가
                if random.random() < 0.30:
                    jong = random.choice(possible_jongs)
            else:
                # 종성이 있는 경우
                if jong == 16:  # ㅁ인 경우 그대로 유지
                    pass
                else:
                    r = random.random()
                    if r < 0.4:
                        # 40% 확률로 원래 종성 유지
                        pass
                    elif r < 0.6:
                        # 20% 확률로 종성 삭제
                        jong = 0
                    else:
                        # 40% 확률로 종성 변경 (규칙 적용)
                        if jong in jong_transformation:
                            jong = random.choice(jong_transformation[jong])
                        # 변환 규칙이 없으면 그대로 유지 (예외 처리)

            new_code = BASE_CODE + (cho * NUM_JUNG + jung) * NUM_JONG + jong
            result_chars.append(chr(new_code))
        else:
            result_chars.append(char)

    return "".join(result_chars)

def transform_sentence(text: str) -> str:
    """
    입력된 문장에 대해 다음 순서로 변형을 적용합니다.
      1. 50% 확률로 apply_liaison을 이용한 연음화 (연음이 가능한 경우)
      2. change_initials: 초성 변환
      3. change_medials: 중성 변환
      4. change_final: 종성 변환
    """
    # 1. 연음 적용 (50% 확률)
    if random.random() < 0.5:
        text = apply_liaison(text)

    # 2. 초성 변환
    text = change_initials(text)

    # 3. 중성 변환
    text = change_medials(text)

    # 4. 종성 변환
    text = change_final(text)

    return text

def final_transform_sentence(text: str) -> str:
    """
    문장을 단어 단위로 순회하며 아래 규칙에 따라 변형합니다.
      1. 한글 정상 음절이 아닌 단일 글자(예: 'ㅎ', 'ㅋ')는 그대로 둡니다.
      2. 영어, 특수부호, 숫자 등 한글 음절이 없는 단어는 변경하지 않습니다.
      3. 각 단어별로 10% 확률로 변형:
         - 50% 확률로 해당 단어 전체에 transform_sentence 함수를 적용.
         - 50% 확률로 단어 내 한 음절을 선택해 임의의 한글 음절(가~힣)로 치환.
      4. 만약 순회 중 변형된 단어가 하나도 없다면, 변형이 가능한 단어 중 하나를 강제로 변형합니다.
    """
    tokens = text.split()  # 공백 기준 단어 토큰화
    transformed_any = False
    new_tokens = []
    
    # 기록: 변형이 가능한 단어의 인덱스 (한글 음절이 하나라도 있는 경우)
    eligible_indices = []
    
    for i, token in enumerate(tokens):
        # 1. 단일 글자이고, 해당 글자가 정상적인 한글 음절(가~힣)이 아닌 경우는 그대로 둠.
        if len(token) == 1 and not (0xAC00 <= ord(token) <= 0xD7A3):
            new_tokens.append(token)
            continue

        # 2. 단어 내에 한글 음절이 하나도 없는 경우 (예: "abc123", "!!")는 그대로 둠.
        if not any(0xAC00 <= ord(ch) <= 0xD7A3 for ch in token):
            new_tokens.append(token)
            continue

        # 해당 토큰은 변형 가능하므로 기록
        eligible_indices.append(i)
        
        # 3. 각 단어별 10% 확률로 변형
        if random.random() < 0.1:
            transformed_any = True
            # 3-1. 50% 확률로 전체 단어에 transform_sentence 적용
            if random.random() < 0.5:
                new_token = transform_sentence(token)
            else:
                # 3-2. 단어 내 한 음절 선택하여 임의의 한글 음절로 치환
                token_chars = list(token)
                # 한글 정상 음절인 문자 인덱스 선택
                indices = [j for j, ch in enumerate(token_chars) if 0xAC00 <= ord(ch) <= 0xD7A3]
                if indices:
                    chosen_index = random.choice(indices)
                    token_chars[chosen_index] = chr(random.randint(0xAC00, 0xD7A3))
                new_token = "".join(token_chars)
            new_tokens.append(new_token)
        else:
            new_tokens.append(token)

    # 4. 한 단어도 변형되지 않았다면, 변형 가능한 단어 중 하나를 강제 변형
    if not transformed_any and eligible_indices:
        # 무작위로 하나의 변형 가능한 토큰 인덱스를 선택
        idx = random.choice(eligible_indices)
        # 강제로 transform_sentence 적용 (전체 단어 변형)
        new_tokens[idx] = transform_sentence(tokens[idx])
    
    return " ".join(new_tokens)


def clean_text(text: str) -> str:
    """
    양쪽 공백 제거 및 단어 사이의 연속 공백을 한 칸으로 변경합니다.
    """
    text = text.strip()
    text = re.sub(r"\s{2,}", " ", text)
    return text

def main():
    df_train = pd.read_csv("../database/train/train.csv")
    num_variants = 10  # 각 output 문장당 생성할 noise 문장의 개수

    rows = []
    for _, row in df_train.iterrows():
        original = clean_text(row["output"])
        # 각 output 문장에 대해 여러 noise 문장을 생성
        for i in range(num_variants):
            noise = final_transform_sentence(original)
            noise = clean_text(noise)
            rows.append({
                "ID": row["ID"],
                "input": noise,
                "output": original
            })

    df_noisy = pd.DataFrame(rows, columns=["ID", "input", "output"])
    output_path = "../database/train/noisy_train.csv"
    df_noisy.to_csv(output_path, index=False)
    print(f"변환 완료, 저장 위치: {output_path}")

if __name__ == "__main__":
    main()

