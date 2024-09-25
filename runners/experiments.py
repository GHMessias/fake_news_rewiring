import torch
from utils.utils import *

def experiments(args):
    # Organizar os dados
    data = text_to_data(args)
    data = organize_data(data, args)
    print(data)
    # treinar os modelos

    # salvar os resultados
    return'oi'