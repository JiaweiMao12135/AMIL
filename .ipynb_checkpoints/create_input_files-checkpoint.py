from utils import create_input_files, train_word2vec_model
import csv
csv.field_size_limit(500*1024*1024)
if __name__ == '__main__':
    create_input_files(csv_folder='data_set/',
                       output_folder='data_file/',
                       sentence_limit=10000,
                       word_limit=200,
                       min_word_count=2)

    train_word2vec_model(data_folder='data_file/',
                         algorithm='skipgram'
                         )

