class PathConfig:
    raw_data_dir = 'raw/'
    gold_data_dir = 'gold/'
    train_sections = set(list(range(2, 23)))
    dev_sections = {0, 1}
    test_sections = {23, 24}