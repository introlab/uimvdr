# GRO722 problématique
# Auteur: Jean-Samuel Lauzon et  Jonathan Vincent
# Hivers 2022
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from models import *

if __name__ == '__main__':

    # ---------------- Paramètres et hyperparamètres ----------------#
    force_cpu = False           # Forcer a utiliser le cpu?
    trainning = True
    visualize_validation = False
    visualize_result = False
    visualize_attention = False
    visualize_confusion = False
    test = True
    learning_curves = True     # Affichage des courbes d'entrainement?
    seed = 42                # Pour répétabilité
    n_workers = 0           # Nombre de threads pour chargement des données (mettre à 0 sur Windows)

    n_hidden = 20  # Nombre de neurones caches par couche
    input_dim = 2  # nombre de feature d'entrée sur l'encodeur
    n_layers = 2  # Nombre de de couches

    batch_size = 64  # Taille des lots
    lr = 8e-4  # Taux d'apprentissage pour l'optimizateur 8e-4
    n_epochs = 200
    train_val_split = .7

    # ---------------- Fin Paramètres et hyperparamètres ----------------#

    # Initialisation des variables
    if seed is not None:
        torch.manual_seed(seed) 
        np.random.seed(seed)

    # Choix du device
    device = torch.device("cuda" if torch.cuda.is_available() and not force_cpu else "cpu")

    # Instanciation de l'ensemble de données
    dataset = HandwrittenWords(r"data_trainval.p")

    # Séparation du dataset (entraînement et validation)
    n_train_samp = int(len(dataset) * train_val_split)
    n_val_samp = len(dataset) - n_train_samp
    dataset_train, dataset_val = torch.utils.data.random_split(dataset, [n_train_samp, n_val_samp])

    # Instanciation des dataloaders
    dataload_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_workers)
    dataload_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_workers)

    # Instanciation du model
    model = trajectory2seq(
        input_dim=input_dim,
        hidden_dim=n_hidden,
        int2symb=dataset.token_to_symbol,
        symb2int=dataset.symbol_to_token,
        device=device,
        max_len={'input': dataset.points_max_len, 'target': dataset.symbols_max_len},
        n_layers=n_layers
    )
    model = model.to(device)

    # Afficher le résumé du model
    print('Model : \n', model, '\n')
    print('Nombre de poids: ', sum([i.numel() for i in model.parameters()]))

    # Initialisation des variables
    best_val_loss = np.inf  # pour sauvegarder le meilleur model

    if trainning:

        criterion = nn.CrossEntropyLoss(ignore_index=dataset.symbol_to_token[dataset.pad_symbol])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        val_loss = []
        train_loss = []
        dist_train = []
        dist_val = []

        # Initialisation affichage
        if learning_curves:
            fig, (ax1, ax2) = plt.subplots(2)  # Initialisation figure

        for epoch in range(1, n_epochs + 1):
            # Entraînement
            running_loss_train = 0
            running_dist_train = 0
            batch_idx = 0
            model.train()

            for batch_target, batch_points in dataload_train:

                batch_target = batch_target.to(device).long()
                batch_points = batch_points.to(device).float()
                batch_size_real = batch_target.shape[0]

                optimizer.zero_grad()

                x, h, attention = model(batch_points, batch_target)
                loss = criterion(x.permute(0, 2, 1), batch_target)

                loss.backward()
                optimizer.step()
                running_loss_train += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(x, dim=-1).detach().cpu().tolist()
                target_seq_list = batch_target.detach().cpu().tolist()

                for i in range(batch_size_real):
                    a = target_seq_list[i]
                    b = output_list[i]
                    M = a.index(1)
                    N = b.index(1) if 1 in b else len(b)
                    running_dist_train += edit_distance(a[:M], b[:N])/batch_size_real

                # Affichage pendant l'entraînement
                if batch_idx % 50 == 0:
                    total_data = batch_idx * batch_size_real
                    percentage_done = 100.0 * batch_idx / len(dataload_train)
                    average_loss = running_loss_train / (batch_idx + 1)

                    print('Train - Epoch: {}/{} [{}/{} ({:.0f}%)] Average Loss: {:.6f}'.format(
                        epoch, n_epochs, total_data, len(dataload_train.dataset), percentage_done, average_loss, end='\r'))
                batch_idx += 1

            # Validation
            running_loss_val = 0
            running_dist_val = 0
            model.eval()

            for batch_target, batch_points in dataload_val:

                batch_target = batch_target.to(device).long()
                batch_points = batch_points.to(device).float()
                batch_size_real = batch_target.shape[0]

                x, h, attention = model(batch_points)
                loss = criterion(x.permute(0, 2, 1), batch_target)
                running_loss_val += loss.item()

                # calcul de la distance d'édition
                output_list = torch.argmax(x, dim=-1).detach().cpu().tolist()
                target_seq_list = batch_target.detach().cpu().tolist()

                for i in range(batch_size_real):
                    a = target_seq_list[i]
                    b = output_list[i]
                    M = a.index(1)
                    N = b.index(1) if 1 in b else len(b)
                    running_dist_val += edit_distance(a[:M], b[:N])/batch_size_real

            current_val_dist = running_dist_val / len(dataload_val)
            print('\nValidation - Average loss: {:.4f} Distance: {:.2f}'.format(running_loss_val/len(dataload_val), current_val_dist))
            print('')

            # Ajouter les loss aux listes
            train_loss.append(running_loss_train / len(dataload_train))
            val_loss.append(running_loss_val / len(dataload_val))
            dist_train.append(running_dist_train / len(dataload_train))
            dist_val.append(current_val_dist)

            # Enregistrer les poids
            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                torch.save(model, 'model.pt')
                print("Saving new best model")

            # Affichage
            if learning_curves:
                ax1.cla()
                ax1.plot(train_loss, label='training loss')
                ax1.plot(val_loss, label='validation loss')
                ax1.legend()
                ax2.cla()
                ax2.plot(dist_train, label="training distance")
                ax2.plot(dist_val, label="validation distance")
                ax2.legend()
                plt.draw()
                plt.pause(0.1)

            # Enregistrer les poids
            if running_loss_val < best_val_loss:
                best_val_loss = running_loss_val
                torch.save(model, 'model.pt')

            # Terminer l'affichage d'entraînement
        if learning_curves:
            plt.show()
            plt.close('all')

    if visualize_validation:
        model = torch.load('model.pt')
        model.eval()

        # Affichage des résultats
        confused_mat = torch.zeros((26, 26))

        for i in range(10):
            # Extraction d'une séquence du dataset de validation
            target_seq, input_points = dataset[np.random.randint(0, len(dataset))]

            # Évaluation de la séquence
            output, hidden, attn = model(input_points[None, :].to(device).float())
            tokenized_output = torch.argmax(output, dim=-1).detach().cpu().numpy().squeeze()
            output = dataset.tokenized_to_str(tokenized_output)

            if visualize_result:
                # Affichage lettre et guess
                plt.plot(input_points[:, 0], input_points[:, 1], label='input sequence')
                plt.title(f"Target word: {dataset.tokenized_to_str(target_seq.detach().cpu().tolist())} | Prediction of network: {output}")
                plt.legend()
                plt.show()

            if visualize_attention:
                # affichage highlight d'attention
                fig, plots = plt.subplots(6)
                attn = attn * 20
                attn[attn > 1] = 1
                for letter in range(attn.shape[2]):

                    plots[letter].scatter(
                        input_points[:, 0].detach().cpu(),
                        input_points[:, 1].detach().cpu(),
                        alpha=attn[0, :, letter].detach().cpu().tolist(),
                        color="green",
                        label='attention'
                    )
                    plots[letter].plot(
                        input_points[:, 0],
                        input_points[:, 1],
                        alpha=0.7,
                        color="black",
                        label='input sequence'
                    )
                    plots[letter].set_ylabel(dataset.token_to_symbol[tokenized_output[letter]], weight='bold', rotation=0)

                plots[0].set_title(dataset.tokenized_to_str(target_seq.detach().cpu().tolist()), weight='bold')
                plt.show()

            if visualize_confusion:
                confused_mat += confusion_matrix(target_seq.int(), tokenized_output)

        if visualize_confusion:
            # 2D plot of the confusion matrix
            plt.figure()
            plt.xticks(np.arange(len(confused_mat)), list(dataset.token_to_symbol.values())[3:], rotation=45)
            plt.yticks(np.arange(len(confused_mat)), list(dataset.token_to_symbol.values())[3:], rotation=45)
            plt.imshow(confused_mat/confused_mat.max(), cmap='plasma', origin='lower', vmax=1, vmin=0)
            plt.show()

            # Surface plot of the confusion matrix
            x = np.linspace(0, dim1 := confused_mat.shape[0], dim1)
            y = np.linspace(0, dim2 := confused_mat.shape[1], dim2)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.xticks(np.arange(len(confused_mat)), list(dataset.token_to_symbol.values())[3:], rotation=45)
            plt.yticks(np.arange(len(confused_mat)), list(dataset.token_to_symbol.values())[3:], rotation=45)
            x_2d, y_2d = np.meshgrid(x, y)
            ax.plot_surface(x_2d, y_2d, np.array(confused_mat / confused_mat.max()), cmap='plasma')
            plt.show()

    if test:
        # Instanciation de l'ensemble de données de test
        test_dataset = HandwrittenWords(r"data_test_no_labels.p")

        # Charger le modèle
        model = torch.load('best_model.pt')
        model.eval()

        # Instanciation de la matrice de confusion vide
        confused_mat = torch.zeros((27, 27))

        for i in range(10):
            random_index = random.randint(0, len(test_dataset)-1)
            target_seq, input_points = test_dataset[random_index]

            # Évaluation de la séquence
            output, hidden, attn = model(input_points[None, :].to(device).float())
            tokenized_output = torch.argmax(output, dim=-1).detach().cpu().numpy().squeeze()
            output = test_dataset.tokenized_to_str(tokenized_output)

            # Affichage des résultats de test
            if visualize_result:
                # Affichage lettre et guess
                plt.plot(input_points[:, 0], input_points[:, 1], label='input sequence')
                plt.title(
                    f"Prediction of network: {output}")
                plt.legend()
                plt.show()

            # Affichage de l'attention
            if visualize_attention:
                # affichage highlight d'attention
                fig, plots = plt.subplots(6)
                attn = attn * 20
                attn[attn > 1] = 1
                for letter in range(attn.shape[2]):
                    plots[letter].scatter(
                        input_points[:, 0].detach().cpu(),
                        input_points[:, 1].detach().cpu(),
                        alpha=attn[0, :, letter].detach().cpu().tolist(),
                        color="green",
                        label='attention'
                    )
                    plots[letter].plot(
                        input_points[:, 0],
                        input_points[:, 1],
                        alpha=0.7,
                        color="black",
                        label='input sequence'
                    )
                    plots[letter].set_ylabel(test_dataset.token_to_symbol[tokenized_output[letter]], weight='bold',
                                             rotation=0)

                plots[0].set_title(test_dataset.tokenized_to_str(target_seq.detach().cpu().tolist()), weight='bold')
                plt.show()

            # Affichage de la matrice de confusion
            if visualize_confusion:
                confused_mat += confusion_matrix(target_seq.int(), tokenized_output)

        if visualize_confusion:
            symbols = []
            symbols.append(test_dataset.token_to_symbol[1])
            symbols.extend((list(test_dataset.token_to_symbol.values())[3:]))
            plt.figure()
            plt.xticks(np.arange(len(confused_mat)), symbols, rotation=45)
            plt.yticks(np.arange(len(confused_mat)), symbols, rotation=45)
            plt.imshow(confused_mat / confused_mat.max(), cmap='plasma', origin='lower', vmax=1, vmin=0)
            plt.show()

            # Surface plot of the confusion matrix
            x = np.linspace(0, dim1 := confused_mat.shape[0], dim1)
            y = np.linspace(0, dim2 := confused_mat.shape[1], dim2)
            fig = plt.figure()
            ax = plt.axes(projection='3d')
            plt.xticks(np.arange(len(confused_mat)), symbols, rotation=45)
            plt.yticks(np.arange(len(confused_mat)), symbols, rotation=45)
            x_2d, y_2d = np.meshgrid(x, y)
            ax.plot_surface(x_2d, y_2d, np.array(confused_mat / confused_mat.max()), cmap='plasma')
            plt.show()
