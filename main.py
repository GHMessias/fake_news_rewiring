'''
Main file of the experiments
'''

from runners.doc2vec import doc2vec_matrix
from utils.arguments import parse_arguments, load_config_from_json
from utils.dataset_transform import process_data

if __name__ == '__main__':
    args = parse_arguments()
    if args.config:
        config_params = load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)
    
    if args.process_data:
        process_data(args.process_input_path, args.process_output_path, args.language)

    if args.train:
        'oi'

    

    



