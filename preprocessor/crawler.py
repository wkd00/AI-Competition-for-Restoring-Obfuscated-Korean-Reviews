import time
import os
import pandas as pd
import multiprocessing
import numpy as np
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

class AirbnbfyCrawler:
    def __init__(self, chrome_driver_path, augmentation_factor=3):
        """크롤러 초기화"""
        self.augmentation_factor = augmentation_factor  # 각 리뷰당 생성할 난독화 문장 개수
        self.chrome_driver_path = chrome_driver_path  # 드라이버 경로

    def _initialize_driver(self):
        """브라우저 드라이버 초기화"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # 백그라운드 실행
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        service = Service(self.chrome_driver_path)
        return webdriver.Chrome(service=service, options=chrome_options)

    def open_site(self, driver):
        """사이트 접속"""
        driver.get("https://airbnbfy.hanmesoft.com/")
        time.sleep(3)  # 페이지 로딩 대기

    def input_sentence(self, driver, sentence):
        """문장을 입력"""
        try:
            input_box = driver.find_element(By.CLASS_NAME, "form-control")
            input_box.clear()
            input_box.send_keys(sentence)  # 문장 입력
        except Exception as e:
            print(f"입력 실패: {sentence}, 오류: {e}")

    def click_convert_button(self, driver):
        """변환 버튼을 클릭하여 난독화된 문장 생성"""
        try:
            convert_button = driver.find_element(By.CLASS_NAME, "btn-primary")
            convert_button.click()
            time.sleep(1)  # 변환 대기

            # 변환된 문장 가져오기
            output_box = driver.find_elements(By.CLASS_NAME, "form-control")[-1]
            return output_box.get_attribute("value").strip()
        except Exception as e:
            print(f"변환 실패, 오류: {e}")
            return None

    def process_batch(self, batch_data, max_existing_id, results_list):
        """데이터 배치를 처리하여 난독화된 문장을 생성"""
        driver = self._initialize_driver()
        self.open_site(driver)
        local_results = []
        
        for _, row in batch_data.iterrows():
            original_text = row["input"]
            self.input_sentence(driver, original_text)

            for j in range(self.augmentation_factor):
                obfuscated_text = self.click_convert_button(driver)
                if obfuscated_text:
                    max_existing_id += 1
                    new_id = f"TRAIN_{max_existing_id:05d}"
                    local_results.append({"input": original_text, "output": obfuscated_text, "ID": new_id})

        results_list.extend(local_results)
        driver.quit()  # 프로세스마다 브라우저 종료

    def run(self, input_file, num_processes=16):
        """멀티프로세싱을 활용한 병렬 크롤링"""
        df = pd.read_csv(input_file)

        # 'input' 및 'ID' 열이 존재하는지 확인
        if "input" not in df.columns or "ID" not in df.columns:
            raise ValueError("CSV 파일에 'input' 또는 'ID' 열이 존재하지 않습니다.")

        # 기존 ID 값 찾기 (TRAIN_000XX 형식 중 가장 큰 값)
        df["ID"] = df["ID"].str.extract(r'(\d+)').astype(int)  # 숫자 부분만 추출
        max_existing_id = df["ID"].max() if not df["ID"].isnull().all() else -1  # 가장 큰 ID 값 찾기

        # 데이터 배치 나누기 (num_processes 개수만큼)
        data_batches = np.array_split(df, num_processes)

        # 멀티프로세싱 실행
        manager = multiprocessing.Manager()
        results_list = manager.list()  # 프로세스 간 공유 가능한 리스트

        processes = []
        for batch in data_batches:
            p = multiprocessing.Process(target=self.process_batch, args=(batch, max_existing_id, results_list))
            processes.append(p)
            p.start()

        # 모든 프로세스 종료 대기
        for p in processes:
            p.join()

        # 결과를 데이터프레임으로 변환
        new_data = pd.DataFrame(list(results_list))

        # 기존 데이터와 새로운 데이터 합치기
        final_df = pd.concat([df.drop(columns=["ID"]), new_data], ignore_index=True)

        # 입력 파일과 같은 디렉토리에 저장할 파일명 설정
        directory = os.path.dirname(input_file)
        base_name = os.path.basename(input_file)

        # 파일명 변경: "augmented_X.csv" → "preprocessed_X.csv"
        if base_name.startswith("augmented_"):
            new_filename = "preprocessed_" + base_name.replace("augmented_", "", 1)
        else:
            new_filename = "preprocessed_" + base_name

        output_path = os.path.join(directory, new_filename)

        # 결과 저장
        final_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        
        print(f"✅ 병렬 크롤링 완료! 변환된 문장이 {output_path}에 저장됨.")
