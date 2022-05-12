import argparse
import numpy as np
from utils import text_helper

def main(input_dirs):
    qst_vocab = text_helper.VocabDict(input_dir+'/vocab_questions.txt')
    
    print(qst_vocab)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_dir', type=str, default='./datasets',
                        help='input directory for visual question answering.')
    
    args = parser.parse_args()
    
    main(args)