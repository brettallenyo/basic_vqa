import argparse
import numpy as np
from utils import text_helper
import random
import spacy
import heapq
import pickle
import warnings
warnings.filterwarnings("ignore")
import logging

nlp = spacy.load('en_core_web_sm')

def main(args):
    qst_vocab = text_helper.VocabDict(args.question_file)
    
    words = []
    
    for word, idx in qst_vocab.word2idx_dict.items():
        words.append(word)
    
    
    random_indices = random.sample(range(63, len(words)), 1023)
    
    new_words = {1 : '<unk>'}
    '''
    
    example1 = random_indices[0]
    word1 = words[example1]
    example2 = random_indices[1]
    word2 = words[example2]
    
    tokens = nlp(word1 + " " + word2)
    for token in tokens:
        # Printing the following attributes of each token.
        # text: the word string, has_vector: if it contains
        # a vector representation in the model, 
        # vector_norm: the algebraic norm of the vector,
        # is_oov: if the word is out of vocabulary.
        print(token.text, token.has_vector, token.vector_norm, token.is_oov)
    
    token1, token2 = tokens[0], tokens[1]
  
    print("Similarity:", token1.similarity(token2))
    '''
    
    # index -> {index_of_special_word -> similarity}
    word_index_to_special_index_to_scores = {}
    
    # heap = [(similarity, index_of_word)]
    special_index_to_heap = {}
    
    def put_word_into_heap(i):
        not_found = True
        next_index = 1
        sorted_scores = sorted(((v, k) for k, v in word_index_to_special_index_to_scores[i].items()))
        while not_found:
            most_similar_special_index = sorted_scores[-next_index][1]
            similarity = sorted_scores[-next_index][0]
            if most_similar_special_index not in special_index_to_heap:
                special_index_to_heap[most_similar_special_index] = []
                heapq.heappush(special_index_to_heap[most_similar_special_index], (similarity, i))
                not_found = False
            elif len(special_index_to_heap[most_similar_special_index]) >= 7 and similarity > special_index_to_heap[most_similar_special_index][0][0]:
                index_to_move = heapq.heappop(special_index_to_heap[most_similar_special_index])[1]
                heapq.heappush(special_index_to_heap[most_similar_special_index], (similarity, i))
                put_word_into_heap(index_to_move)
                not_found = False
            elif len(special_index_to_heap[most_similar_special_index]) < 7:
                heapq.heappush(special_index_to_heap[most_similar_special_index], (similarity, i))
                not_found = False
            next_index += 1
    
    for i in range(63, len(words)):
        if i in random_indices:
            continue
        word_index_to_special_index_to_scores[i] = {}
        word = words[i]
        for special_index in random_indices:
            other_word = words[special_index]
            phrase = word + " " + other_word
            tokens = nlp(phrase)
            token1, token2 = tokens[0], tokens[1]
            similarity = token1.similarity(token2)
            word_index_to_special_index_to_scores[i][special_index] = similarity
        put_word_into_heap(i)
        logging.info(special_index_to_heap)
        logging.info(i)
        print(i)
    
    with open(args.pickle_file, 'wb') as handle:
        pickle.dump(special_index_to_heap, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    # get rid of pad
    # 0 = unk
    # the next 60 items should be grouped together as punctuation
    
    
    
        
    
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--question_file', type=str, default='/content/data/vocab_questions.txt',
                        help='input directory for visual question answering.')
    parser.add_argument('--pickle_file', type=str, default='/content/data/vocab_questions.txt',
                        help='input directory for visual question answering.')
    parser.add_argument('--log_file', type=str, default='/content/data/vocab_questions.txt',
                        help='input directory for visual question answering.')
    
    args = parser.parse_args()
    
    logging.basicConfig(filename=args.log_file, level=logging.DEBUG)
    
    main(args)