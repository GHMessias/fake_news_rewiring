import torch
from utils.utils import *

def train_gae(data, gae_model, optimizer, epochs, verbose = False):
    if isinstance(gae_model.encoder, RGCN):
        for e in range(epochs):
            optimizer.zero_grad()
            H_L = gae_model.encode(data.x.float(), data.graph_list)
            loss = gae_model.recon_loss(H_L, data.graph_list[-1].edge_index)
            if verbose:
                print(f'epoch {e+1} | loss {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()
    if isinstance(gae_model.encoder, GCN):
        for e in range(epochs):
            optimizer.zero_grad()
            H_L = gae_model.encode(data.x.float(), data.edge_index)
            loss = gae_model.recon_loss(H_L, data.edge_index)
            if verbose:
                print(f'epoch {e+1} | loss {loss.item()}', end = '\r')
            loss.backward()
            optimizer.step()
    print('\n')
    return

def experiments(args):
    # Organizar os dados
    data = text_to_data(args)
    data = organize_data(data, args)
    print(data.x[0])
    print(data.graph_list)

    # treinar os modelos
    for model_name in args.models:
        model = get_model(model_name, data, args)
        print(model)

        if model_name in ['RGCN', 'GCN']:
            optimizer = torch.optim.Adam(params=model.parameters(), lr = 0.001) 
            train_gae(data, model, optimizer, epochs = args.epochs_gae, verbose = True)
                
             # salvar os resultados