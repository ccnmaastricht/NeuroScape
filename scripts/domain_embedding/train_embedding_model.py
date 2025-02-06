import os
import time
import torch

import pandas as pd
import numpy as np

from collections import deque

from src.utils.domain_embedding import *
from src.utils.parsing import parse_directories
from src.utils.load_and_save import load_embedding_shards

from src.classes.info_nce_loss import InfoNCELoss
from src.classes.sparse_embedding_network import SparseEmbeddingNetwork

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']


def train_model(model, train_data, validation_data, validation_window,
                grokfast_ema_params, infoNCE_loss_function, correlation_weight,
                epochs, batch_size, device, optimizer, scheduler,
                model_save_directory):
    """
    Train the model.

    Parameters:
    - model: PyTorch model, the model to train.
    - train_data: Tensor, the training data.
    - validation_data: Tensor, the validation data.
    - grokfast_ema_params: Dict, the Grokfast EMA parameters.
    - infoNCE_loss_function: InfoNCELoss, the InfoNCE loss function.
    - correlation_weight: Float, the weight of the correlation loss.
    - epochs: Integer, the number of epochs.
    - batch_size: Integer, the batch size.
    - device: PyTorch device, the device to use.
    - optimizer: PyTorch optimizer, the optimizer to use.
    - scheduler: PyTorch scheduler, the scheduler to use.
    - model_save_directory: String, the path to save the model.
    """

    start = time.time()

    best_loss = validate(model, validation_data, infoNCE_loss_function)
    validation_losses = deque([best_loss] * validation_window)
    logging = {
        'epoch': [],
        'training loss': [],
        'validation loss': [],
    }
    current_model_file = os.path.join(model_save_directory,
                                      'domain_embedding_model_current.pth')

    logging_file = os.path.join(model_save_directory, 'logging.csv')

    # Load Current Model and logging
    if os.path.exists(current_model_file):
        print('Continuing training from previous model')
        model.load_state_dict(torch.load(current_model_file))
        logging = load_log(logging_file)
        best_loss = min(logging['validation loss'])
    else:
        print('Starting training from scratch')

    # Initialize Gradients for EMA
    gradients = None
    total_samples = train_data.shape[0]

    numerator = total_samples // batch_size
    for epoch in range(epochs):
        train_loss_average = 0
        correlation_loss_average = 0

        np.random.shuffle(train_data)

        for j in range(0, total_samples, batch_size):
            indices = range(j, min(j + batch_size, total_samples))
            X_batch = extract_batch(train_data, indices, device)
            train_loss, correlation_loss, gradients = train_one_batch(
                model, X_batch, infoNCE_loss_function, correlation_weight,
                optimizer, gradients, grokfast_ema_params)
            train_loss_average += train_loss
            correlation_loss_average += correlation_loss

        train_loss_average /= numerator
        correlation_loss /= numerator
        correlation_loss /= correlation_weight
        # Step the scheduler
        scheduler.step()

        # Validate Model
        current_val_loss = validate(model, validation_data,
                                    infoNCE_loss_function)

        # Update Validation Loss
        validation_losses.popleft()
        validation_losses.append(current_val_loss)

        validation_loss = sum(validation_losses) / validation_window

        # Report Results
        print(
            f'Epoch {epoch + 1:5d}/{epochs}, Training Loss: {train_loss_average:.4f}, Validation Loss: {validation_loss:.4f}, Correlation Loss: {correlation_loss:.4f}, Time Taken: {time.time() - start:.2f}s'
        )
        print('---' * 10)

        # Save logging
        logging['epoch'].append(epoch + 1)
        logging['training loss'].append(train_loss_average)
        logging['validation loss'].append(validation_loss)
        pd.DataFrame(logging).to_csv(logging_file, index=False)

        # Save Current Model
        save_model(model,
                   model_save_directory,
                   name='domain_embedding_model_current.pth')

        # Reset Timer
        start = time.time()

        # Save Best Model
        if validation_loss < best_loss:
            best_loss = validation_loss
            save_model(model,
                       model_save_directory,
                       name='domain_embedding_model_best.pth')


if __name__ == '__main__':

    # Load Configurations
    configurations = load_configurations()
    directories = parse_directories()

    data_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['hdf5']['voyage'])
    model_save_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['models'])
    train_data_file = os.path.join(data_directory, 'train_data.txt')
    validation_data_file = os.path.join(data_directory, 'validation_data.txt')

    device = configurations['model']['device']
    input_dimension = configurations['model']['input_dimension']
    hidden_dimensions = configurations['model']['hidden_dimensions']
    output_dimension = configurations['model']['output_dimension']

    epochs = configurations['training']['epochs']
    batch_size = configurations['training']['batch_size']

    initial_learning_rate = configurations['training']['initial_learning_rate']
    minimum_learning_rate = configurations['training']['minimum_learning_rate']
    gamma = (minimum_learning_rate / initial_learning_rate)**(1 / epochs)
    validation_window = configurations['training']['validation_window']
    validation_size = configurations['training']['validation_size']

    grokfast_ema_params = configurations['grokfast_ema']

    dropout = configurations['regularization']['dropout']
    l2_weight = configurations['regularization']['l2_weight']
    correlation_weight = configurations['regularization']['correlation_weight']

    info_nce_temperature = configurations['InfoNCE']['temperature']
    cutoff_values = configurations['InfoNCE']['cutoff_values']

    # Define Model
    embedding_network = SparseEmbeddingNetwork(
        input_dimension=input_dimension,
        hidden_dimensions=hidden_dimensions,
        output_dimension=output_dimension,
        dropout=dropout).to(device)

    # Define Loss
    infoNCE_loss_function = InfoNCELoss(temperature=info_nce_temperature,
                                        cutoff_values=cutoff_values)

    # Get Optimizer and Scheduler
    optimizer, scheduler = setup_optimizer_scheduler(embedding_network,
                                                     initial_learning_rate,
                                                     gamma, l2_weight)

    # Get Data
    if os.path.exists(train_data_file):
        print('Loading training and validation data from files')
        with open(train_data_file, 'r') as f:
            train_files = f.read().splitlines()
        with open(validation_data_file, 'r') as f:
            validation_files = f.read().splitlines()

    else:
        print('Splitting data into training and validation sets')
        train_files, validation_files = get_and_split_files(
            data_directory, validation_size)

        print(
            f'\tTraining Files: {len(train_files)}, Validation Files: {len(validation_files)}'
        )

        with open(train_data_file, 'w') as f:
            f.write('\n'.join(train_files))
        with open(validation_data_file, 'w') as f:
            f.write('\n'.join(validation_files))

    validation_data, _ = load_embedding_shards(validation_files)
    validation_data = torch.tensor(validation_data,
                                   dtype=torch.float32).to(device)

    training_data, _ = load_embedding_shards(train_files)

    # Train Model
    train_model(embedding_network, training_data, validation_data,
                validation_window, grokfast_ema_params, infoNCE_loss_function,
                correlation_weight, epochs, batch_size, device, optimizer,
                scheduler, model_save_directory)

    # Save Final Model
    save_model(embedding_network,
               model_save_directory,
               name='domain_embedding_model_final.pth')
