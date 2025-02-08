from crawler import AirbnbfyCrawler
from augmentator import augment_reviews

# if __name__ == "__main__":
#     input_path = "../database/train/train.csv"
#     augment_reviews(input_path, 2)

#     augmented_path = "../database/train/augmented_train.csv"
#     CHROME_DRIVER_PATH = "chromedriver"
#     preprocessor = AirbnbfyCrawler(CHROME_DRIVER_PATH, augmentation_factor=4)
#     preprocessor.run(augmented_path)

#테스트용
if __name__ == "__main__":
    input_path = "../database/train/dummy/traintest.csv"
    augment_reviews(input_path, 2)

    augmented_path = "../database/train/dummy/augmented_traintest.csv"
    CHROME_DRIVER_PATH = "chromedriver"
    preprocessor = AirbnbfyCrawler(CHROME_DRIVER_PATH, augmentation_factor=4)
    preprocessor.run(augmented_path)

