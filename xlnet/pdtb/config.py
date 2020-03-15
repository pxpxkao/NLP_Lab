class PathConfig:
    raw_data_dir = '../data/PDTB-3.0/data/raw/'
    gold_data_dir = '../data/PDTB-3.0/data/gold/'
    train_sections = set(list(range(2, 23)))
    dev_sections = {0, 1}
    test_sections = {23, 24}