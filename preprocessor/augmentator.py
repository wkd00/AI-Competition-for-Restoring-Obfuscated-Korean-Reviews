import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re
from tqdm import tqdm  # 진행 상태 표시

def augment_reviews(input_path, augmentation_factor=2, batch_size=16):
    """
    주어진 CSV 파일의 'input' 열을 deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 모델을 활용하여 배치 단위로 증강합니다.
    기존 데이터의 마지막 ID에서 이어서 새로운 데이터를 추가합니다.
    
    :param input_path: 원본 리뷰 데이터가 포함된 CSV 파일 경로
    :param augmentation_factor: 각 리뷰당 생성할 유사 리뷰 개수 (기본값: 2)
    :param batch_size: 한 번에 처리할 문장의 개수 (기본값: 4)
    """

    # CUDA 필수 사용: CUDA가 없으면 실행 중지
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA가 필요합니다. GPU에서 실행해주세요.")

    print("CUDA 사용 가능: GPU에서 실행합니다.")

    # 모델 및 토크나이저 로드
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    ).to("cuda")  # 반드시 CUDA에서 실행

    # CSV 파일 읽기
    df = pd.read_csv(input_path)
    
    # 'input' 및 'ID' 열이 존재하는지 확인
    if "input" not in df.columns or "ID" not in df.columns:
        raise ValueError("CSV 파일에 'input' 또는 'ID' 열이 존재하지 않습니다.")
    
    augmented_data = []

    # 기존 ID에서 숫자 부분 추출
    df["ID"] = df["ID"].astype(str)
    df["ID_num"] = df["ID"].apply(lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else -1)
    
    # 가장 큰 ID 값 찾기
    max_existing_id = df["ID_num"].max() if not df["ID_num"].isnull().all() else -1

    # 리뷰 리스트 준비
    review_list = df["input"].tolist()

    # Batch 단위로 모델 실행
    for i in tqdm(range(0, len(review_list), batch_size), desc="Processing Batches"):
        batch_texts = review_list[i : i + batch_size]  # 배치 크기만큼 가져오기
        batch_prompts = [f"다음 문장과 유사한 한국어 리뷰를 생성하세요: {text}\n새로운 리뷰:" for text in batch_texts]

        for _ in range(augmentation_factor):  # 각 문장당 augmentation_factor 만큼 생성
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")

            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=50, temperature=0.7)

            # 각 문장의 생성 결과 저장
            for j, output in enumerate(outputs):
                generated_review = tokenizer.decode(output, skip_special_tokens=True).split("새로운 리뷰:")[-1].strip()
                
                # 새로운 ID 생성
                max_existing_id += 1
                new_id = f"TRAIN_{max_existing_id:05d}"

                augmented_data.append({"input": batch_texts[j], "output": generated_review, "ID": new_id})

    # 결과를 데이터프레임으로 변환
    augmented_df = pd.DataFrame(augmented_data)

    # 기존 데이터와 새로운 데이터 합치기
    final_df = pd.concat([df.drop(columns=["ID_num"]), augmented_df], ignore_index=True)

    # 원본 파일과 동일한 디렉토리에 저장할 파일명 설정
    directory = os.path.dirname(input_path)  # 원본 파일이 있는 폴더 경로
    base_name = os.path.basename(input_path)  # 파일명 추출

    # 파일명 변경: "augmented_X.csv" → "preprocessed_X.csv"
    if base_name.startswith("augmented_"):
        new_filename = "preprocessed_" + base_name.replace("augmented_", "", 1)
    else:
        new_filename = "preprocessed_" + base_name

    output_path = os.path.join(directory, new_filename)  # 저장 경로 설정

    # 결과 저장
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ 리뷰 데이터 증강 완료! 저장 경로: {output_path}")
