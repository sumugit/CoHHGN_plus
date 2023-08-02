import os

class Config:
    age_range = "20_35"
    SEED = 42
    cuda = 0
    BASE_PATH = "/workspace/datasets"
    RAW = os.path.join(BASE_PATH, "raw")
    CLEAN = os.path.join(BASE_PATH, "clean", age_range)
    JSON = os.path.join(BASE_PATH, "json", age_range)
    JSON2 = os.path.join(BASE_PATH, "json2", age_range)
    PROCESSED = os.path.join(BASE_PATH, "processed", age_range)
    raw_path_2019 = os.path.join(RAW, "_".join(["2019", age_range + ".csv"]))
    raw_path_2020 = os.path.join(RAW, "_".join(["2020", age_range + ".csv"]))
    clean_path = os.path.join(CLEAN, age_range + ".csv")
    clean_session_path = os.path.join(CLEAN, "_".join(["session", age_range + ".csv"]))
    category_path = os.path.join(JSON, "_".join(["category_count", age_range + ".json"]))
    gender_path = os.path.join(BASE_PATH, "json", "user_gender.json")
    region_path = os.path.join(BASE_PATH, "json", "region.json")
    category_id_path = os.path.join(JSON, "_".join(["category_to_id", age_range + ".json"]))
    big_category_id_path = os.path.join(JSON, "_".join(["big_category_to_id", age_range + ".json"]))
    middle_category_id_path = os.path.join(JSON, "_".join(["middle_category_to_id", age_range + ".json"]))
    processed_labels_path = os.path.join(PROCESSED, "_".join(["labels", age_range + ".csv"]))
    processed_path = os.path.join(PROCESSED, age_range + ".csv")
    OBJECT = os.path.join(BASE_PATH, "object", age_range)
    train_path = os.path.join(OBJECT, "rk_train.txt")
    test_path = os.path.join(OBJECT, "rk_test.txt")
    srgnn_train_path = os.path.join(OBJECT, "sr_train.txt")
    srgnn_test_path = os.path.join(OBJECT, "sr_test.txt")
    srgnn_all_train_seq_path = os.path.join(OBJECT, "srgnn_all_train_seq.txt")
    all_train_seq_path = os.path.join(OBJECT, "all_train_seq.txt")
    all_test_seq_path = os.path.join(OBJECT, "all_test_seq.txt")
    all_train_price_seq_path = os.path.join(OBJECT, "all_train_price_seq.txt")
    all_test_price_seq_path = os.path.join(OBJECT, "all_test_price_seq.txt")
    item_id_to_node_id_path = os.path.join(JSON, "_".join(["item_id_to_node_id", age_range + ".json"]))
    srgnn_item_id_to_node_id_path = os.path.join(JSON, "_".join(["srgnn_item_id_to_node_id", age_range + ".json"]))