import os
import pandas as pd
import numpy as np
from pdtb.config import PathConfig
from pdtb.dataset import load_pipe_file

def pdtb_preprocess(sections, data_dir, set_type):
    # sections = os.listdir(PathConfig.gold_data_dir)
    instances = []
    for section in sections:
        raw_sec_dir = os.path.join(data_dir, PathConfig.raw_data_dir, section)
        gold_sec_dir = os.path.join(data_dir, PathConfig.gold_data_dir, section)
        if not os.path.isdir(gold_sec_dir):
            continue
        # print(os.listdir(gold_sec_dir))
        for file in os.listdir(gold_sec_dir):
            # print(file)
            raw_path = os.path.join(raw_sec_dir, file)
            gold_path = os.path.join(gold_sec_dir, file)
            pipe_instances = load_pipe_file(raw_path, gold_path)
            # pipe_instances[0].print_inst()
            instances += pipe_instances
    examples = []
    for idx, instance in enumerate(instances):
        examples.append([set_type+"-"+str(idx), instance.arg1, instance.arg2, instance.label])
    df = pd.DataFrame(examples, columns=['Guid', 'Arg1', 'Arg2', 'Label'])
    df.to_csv(os.path.join(data_dir, set_type+'.csv'), ';', index=0)
    print("Total instances:", len(instances))
    return instances

if __name__ == '__main__':
    print('Loading dataset...')
    train_sections = ['{:02}'.format(section_num) for section_num in PathConfig.train_sections]
    dev_sections = ['{:02}'.format(section_num) for section_num in PathConfig.dev_sections]
    test_sections = ['{:02}'.format(section_num) for section_num in PathConfig.test_sections]

    instances = pdtb_preprocess(train_sections, './data', "train")
    instances = pdtb_preprocess(dev_sections, './data', "dev")
    instances = pdtb_preprocess(test_sections, './data', "test")

