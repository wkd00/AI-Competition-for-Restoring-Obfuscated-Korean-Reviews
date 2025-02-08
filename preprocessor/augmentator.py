import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import re
from tqdm import tqdm  # Progress bar

def augment_reviews(input_path, augmentation_factor=2, batch_size=16):
    """
    Augments Korean review data using 'beomi/gemma-ko-7b' model.
    New data consists of an empty 'input' column, a newly generated 'output' column, and a sequential 'ID' column.
    
    :param input_path: Path to the original CSV file containing review data.
    :param augmentation_factor: Number of new similar reviews to generate per existing review (default: 2).
    :param batch_size: Number of reviews to process per batch (default: 16).
    """

    # Ensure CUDA is available
    if not torch.cuda.is_available():
        raise RuntimeError("🚨 CUDA is required. Please run on a GPU-enabled device.")

    print("✅ CUDA detected: Running on GPU.")

    # Load model and tokenizer
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # 모델 로드 (양자화 + Flash Attention 2)
    model = AutoModelForCausalLM.from_pretrained(
        model_name
    )
    model.to("cuda")


    # Load CSV file
    try:
        df = pd.read_csv(input_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(input_path, encoding="cp949")

    # Ensure required columns exist
    if "output" not in df.columns or "ID" not in df.columns:
        raise ValueError("🚨 CSV file must contain 'output' and 'ID' columns.")

    augmented_data = []

    # Extract numerical part of existing IDs
    df["ID"] = df["ID"].astype(str)
    df["ID_num"] = df["ID"].apply(lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else -1)
    
    # Find the last ID value
    max_id = df["ID_num"].max() if not df["ID_num"].isnull().all() else -1

    # Prepare the list of reviews to augment
    review_list = df["output"].tolist()

    # Process in batches
    for i in tqdm(range(0, len(review_list), batch_size), desc="🔄 Processing batches"):
        batch_texts = review_list[i : i + batch_size]  # Get batch
        batch_prompts = [
            f"""숙박시설 리뷰입니다. {review}
            """
            for review in batch_texts
            ]

        for _ in range(augmentation_factor):  # Generate multiple outputs per review
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens = 256, do_sample = True, temperature = 0.7
                    , top_p = 0.9
                    )

            # Store generated results
            for j, output in enumerate(outputs):
                # 기존 코드에서는 아래와 같이 "그리고:"를 기준으로 분리하고 있었음:
                # generated_review = tokenizer.decode(output, skip_special_tokens=True).split("그리고:")[-1].strip()

                # 프롬프트(입력 리뷰) 이후의 텍스트만 출력하고 싶다면, 프롬프트의 길이에 맞춰 잘라낼 수 있습니다.
                decoded_output = tokenizer.decode(output, skip_special_tokens=True)
                prompt_text = batch_prompts[j]
                # 만약 생성 결과가 프롬프트로 시작한다면, 프롬프트 부분을 잘라냅니다.
                if decoded_output.startswith(prompt_text):
                    generated_review = decoded_output[len(prompt_text):].strip()
                else:
                    generated_review = decoded_output.strip()

                # Generate new ID
                max_id += 1
                new_id = f"TRAIN_{max_id:05d}"

                augmented_data.append({ "ID": new_id, "input": "", "output": generated_review})

    # Convert results to DataFrame
    augmented_df = pd.DataFrame(augmented_data)

    # Combine with the original dataset
    final_df = pd.concat([df.drop(columns=["ID_num"]), augmented_df], ignore_index=True)

    # Generate new filename
    folder_path = os.path.dirname(input_path)
    base_filename = os.path.basename(input_path)

    # Modify filename: "train.csv" → "augmented_train.csv"
    if base_filename.endswith(".csv"):
        file_name, ext = os.path.splitext(base_filename)
        new_filename = f"augmented_{file_name}.csv"
    else:
        new_filename = f"augmented_{base_filename}"

    output_path = os.path.join(folder_path, new_filename)

    # Save final dataset
    final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    
    print(f"✅ Review data augmentation complete! Saved to: {output_path}")
