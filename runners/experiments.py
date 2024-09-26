import torch
from utils.utils import *
import pandas as pd
from sklearn import svm
from models.inference_models.models import ExtendedGCN, ExtendedRGCN
from torch_geometric.utils import to_networkx
from networkx.algorithms import node_classification

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

def freeze_model_params(model):
    for param in model.parameters():
        param.requires_grad = False

def gae_negative_inference(data, model, num_neg):
    inference_dict = dict()
    model.eval()
    if isinstance(model.encoder, RGCN):
        H_L = model.encode(data.x, data.graph_list)

    if isinstance(model.encoder, GCN):
        H_L = model.encode(data.x.float(), data.edge_index)

    for element in data.U:
        dist = torch.cdist(H_L[element].unsqueeze(0), H_L[data.P])
        value = dist.mean()
        inference_dict[element] = value
    dicionario_ordenado = dict(sorted(inference_dict.items(),reverse=True, key=lambda item: item[1]))
    return torch.stack(list(dicionario_ordenado.keys())[:num_neg])

def pu_classification(data, model):
    '''
    Função responsável por treinar os modelos a partir da segunda etapa, os dados negativos em data.N representam os elementos inferidos. 
    '''
    for element in data.N:
        data.infered_y[element] = 0

    # weights_gcn = torch.tensor([ len(data.y[data.y == 1]) / len(data.y), len(data.y[data.y == 0])/ len(data.y)])
    # weights_rgcn = torch.tensor([ len(data.y[data.y == 1]) / len(data.y),  len(data.y[data.y == 0])/ (len(data.y))])
    data.train_mask = torch.tensor([1 if data.infered_y[x] in [0,1] else 0 for x in range(data.num_nodes)], dtype = torch.bool)
    if isinstance(model, (CCRNE, MCLS, RCSVM)):
        clf = svm.SVC()
        try:
            clf.fit(data.x[data.train_mask].detach().numpy(), data.infered_y[data.train_mask].detach().numpy())
            y_pred = clf.predict(data.x[data.test_mask])
        except:
            return pd.DataFrame()
        return pd.DataFrame(evaluate(data.y[data.test_mask].detach().numpy(), y_pred, pos_label=1))
        # evaluate(data.y[data.test_mask].detach().numpy(), y_pred, pos_label=1)

    if isinstance(model, GAE):
        if isinstance(model.encoder, RGCN):
            freeze_model_params(model.encoder)
            RGCN_classifier = ExtendedRGCN(model.encoder, 2, output_activation_function=torch.softmax)
            optimizer = torch.optim.Adam(RGCN_classifier.parameters(), lr = 0.01)
            
            criterion = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')

            for epoch in range(200):
                optimizer.zero_grad()
                out = RGCN_classifier(data.x, data.graph_list)
                loss = criterion(out[data.train_mask], data.infered_y[data.train_mask])
                print(f'loss for pu classification: epoch {epoch} | loss {loss.item():.4f}', end = '\r')
                loss.backward()
                optimizer.step()

            RGCN_classifier.eval()
            y_pred = RGCN_classifier(data.x, data.graph_list)[data.test_mask]
            y_pred = torch.argmax(y_pred, dim = 1)
            return pd.DataFrame(evaluate(data.y[data.test_mask].detach().numpy(), y_pred.detach().numpy(), pos_label = 1))
            # evaluate(data.y[data.test_mask].detach().numpy(), y_pred, pos_label=1)
        
        if isinstance(model.encoder, GCN):
            freeze_model_params(model.encoder)
            GCN_classifier = ExtendedGCN(model.encoder, 2).float()
            optimizer = torch.optim.Adam(GCN_classifier.parameters(), lr = 0.01)
            criterion = torch.nn.CrossEntropyLoss(weight = None, reduction='mean')

            for epoch in range(200):
                optimizer.zero_grad()
                out = GCN_classifier(x = data.x, edge_index = data.edge_index)
                loss = criterion(out[data.train_mask], data.infered_y[data.train_mask])
                print(f'loss for pu classification: epoch {epoch} | loss {loss.item():.4f}', end = '\r')
                loss.backward()
                optimizer.step()

            GCN_classifier.eval()
            y_pred = GCN_classifier(data.x, data.edge_index)[data.test_mask]
            y_pred = torch.argmax(y_pred, dim = 1)
            return pd.DataFrame(evaluate(data.y[data.test_mask].detach().numpy(), y_pred.detach().numpy(), pos_label = 1))
            # evaluate(data.y[data.test_mask].detach().numpy(), y_pred, pos_label=1)

    if isinstance(model, (LP_PUL, PU_LP)):
        G = to_networkx(data, to_undirected=True)
        labels = {node: label.item() for node, label in zip(G.nodes(), data.infered_y) if label in [0, 1]}
        nx.set_node_attributes(G, labels, 'label')
        y_pred = torch.tensor(node_classification.local_and_global_consistency(G))
        return pd.DataFrame(evaluate(data.y[data.test_mask].detach().numpy(), y_pred[data.test_mask].detach().numpy()))
        # evaluate(data.y[data.test_mask].detach().numpy(), y_pred[data.test_mask].detach().numpy())

    if isinstance(model, OCSVM):
        model.train()
        y_pred = model.predict()
        y_pred = np.where(y_pred == -1, 0, y_pred)
        print(np.unique(y_pred))
        return pd.DataFrame(evaluate(data.y[data.test_mask].detach().numpy(), y_pred[data.test_mask]))

def experiments(args):
    # Organizar os dados
    df_pu_classify = pd.DataFrame()
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
            data.N = gae_negative_inference(data, model, num_neg = len(data.P))
                
        if model_name in ["RCSVM", "CCRNE", "MCLS", "PU_LP", "LP_PUL"]:
            model.train()
            data.N = model.negative_inference(num_neg = len(data.P))
        
        # salvar os resultados
        df_1 = pu_classification(data, model)
        df_1['model'] = model_name
        df_1['dataset'] = data.name
        df_1['rate'] = args.rate
        df_1['length negatives'] = len(data.N)
        df_1['length positives'] = len(data.P)
        df_pu_classify = pd.concat([df_pu_classify, df_1], ignore_index=True)
        
        df_pu_classify.to_csv(f'results/pu_classify_results_{data.name}.csv')