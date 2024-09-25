import argparse
import json

def parse_arguments():
    '''
    Function to collect the arguments
    '''

    parser = argparse.ArgumentParser()

    # json config
    parser.add_argument('-cnf', '--config', type = str, help = 'Path to json configuration file')

    # process data config
    parser.add_argument('--process_data', action = 'store_true', help = "Process the data removing stopwords and punctuation")
    parser.add_argument('--process_input_path', type = str, default = 'Process data input path (dataset raw)')
    parser.add_argument('--process_output_path', type = str, help = 'Process data output path (processed data)')
    parser.add_argument('--language', type = str, default = 'portuguese')

    # doc2vec config
    parser.add_argument("--vector_size", type = int, help = "Vector size of doc2vec model", default = 500)
    parser.add_argument("--window", type = int, help = "The maximum distance between the current and predicted word within a sentence", default = 8)
    parser.add_argument("--alpha", type = float, default = 0.025, help = "Learning rate of doc2vec model")
    parser.add_argument("--min_alpha", type = float, default = 0.001, help = "Learning rate will linearly drop to min_alpha as training progresses in doc2vec model.")
    parser.add_argument("--epochs", type = int, default = 100, help = "Number of epochs to do2vec model to train")
    parser.add_argument("--dm_mean", type = int, default = 1, help = "Parameter to determine if do2cvec model will use mean")
    parser.add_argument("--dm_concat", type = int, default = 0, help = "Parameter to determine if do2cvec model will use concat")
    parser.add_argument("--dm", type = int, default = 1, help = "Defines the training algorithm. If dm=1, distributed memory (PV-DM) is used")
    parser.add_argument("--load", type = bool, default = False, help = "Parameter to determine if the doc2vec model will be loaded. If set to false, the model will be trained")
    parser.add_argument("--save_model", type = bool, default = True, help = "Parameter to determine if the model will be saved")
    parser.add_argument("--save_model_path", type = str, help = "Directory to save the model")

    return parser.parse_args()

def load_config_from_json(json_file):
    '''Function to load parameters from a JSON file'''
    with open(json_file, 'r') as f:
        config = json.load(f)
    return config

