'''
Main file of the experiments
'''

from utils.arguments import parse_arguments, load_config_from_json
from utils.dataset_transform import process_data
from runners.experiments import experiments
from utils.bibliotecas import *

if __name__ == '__main__':
    args = parse_arguments()
    if args.config:
        config_params = load_config_from_json(args.config)
        # Atualiza os parâmetros do argparse com os valores do JSON
        for key, value in config_params.items():
            setattr(args, key, value)
    
    # if args.process_data:
    #     process_data(args.process_input_path, args.process_output_path, args.language)

    if args.run_experiments:
        experiments(args)
        
if __name__ == '__main__':
    doc2vec_rep = load_representation("datasets-Projeto/datasets/complete datasets/FakeBr/D2V representations/Doc2Vec_model=both_method=concat_dim_size=500_num_max_epochs=1000_window_size=8_num_threads=4_min_count=1_alpha=0.025_min_alpha=0.0001.rep")
    x = doc2vec_rep.text_vectors #acesso as representações 
    y = np.load("datasets-Projeto/datasets/complete datasets/FakeBr/Dataset/labels.npy")