import argparse
from solver import Solver

def parse():
    parser = argparse.ArgumentParser(description="tree transformer")
    parser.add_argument('-model_dir',default='train_model',help='output model weight dir')
    parser.add_argument('-batch_size', type=int, default=64, help='batch size')
    parser.add_argument('-load', default='./train_model/model.pth', help= 'load: model_dir', dest= 'load_model')
    parser.add_argument('-train', action='store_true',help='whether train whole model')
    parser.add_argument('-test', action='store_true',help='whether test')
    parser.add_argument('-train_file',default='./data/task2.train.src',help='training src file')
    parser.add_argument('-train_tgt_file',default='./data/task2.train.tgt',help='training tgt file')
    parser.add_argument('-valid_file',default='./data/task2.val.src',help='dev src file')
    parser.add_argument('-valid_tgt_file',default='./data/task2.val.tgt',help='dev tgt file')
    parser.add_argument('-test_file',default='./data/task2.test.src',help='testing src file')
    parser.add_argument('-test_tgt_file',default='./data/task2.test.tgt',help='testing tgt file')
    parser.add_argument('-pred_dir', default='./pred_dir/', help='prediction dir', dest='pred_dir')
    parser.add_argument('-filename', default='pred.txt', help='prediction file', dest='filename')
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    args = parse()
    solver = Solver(args)
    
    if args.train:
        solver.train()
    elif args.test:
        solver._test()