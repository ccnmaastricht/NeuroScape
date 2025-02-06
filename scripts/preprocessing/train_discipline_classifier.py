import os
import glob
import torch
import numpy as np
import torch.nn as nn

from copy import deepcopy

from src.utils.parsing import parse_directories
from src.classes.discipline_classifier import DisciplineClassifier
from src.utils.classifier import load_configurations, train_one_epoch, validate, \
    save_model, data_loader, to_device, compute_expected_accuracy, compute_kappa, \
    drop_class

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
BASEPATH = os.environ['BASEPATH']


def train_model(model, filter_model, data_directory, model_directory,
                model_name, configurations, device, loss_function, optimizer):
    """
    Train the model.

    Parameters:
    - model: PyTorch model
    - pretrained_model: PyTorch model, the pretrained model.
    - data_directory: str, the path to the data directory.
    - model_directory: str, the path to the model directory.
    - model_name: str, the name of the model.
    - configurations: dict, the configurations.
    - device: PyTorch device.
    - loss_function: PyTorch loss function.
    - optimizer: PyTorch optimizer.

    Returns:
    - best_model: PyTorch model, the best model in terms of validation loss.
    """
    confidence_cutoff = configurations['confidence_cutoff']
    epochs = configurations['epochs']
    buffer_size = configurations['buffer_size']
    batch_size = configurations['batch_size']
    save_path = os.path.join(model_directory, f'{model_name}.pth')
    best_loss = float('inf')
    best_model = None

    train_files = glob.glob(os.path.join(data_directory, 'Train/*.pkl'))
    val_files = glob.glob(os.path.join(data_directory, 'Val/*.pkl'))
    X_val, Y_val = data_loader(val_files)

    if filter_model is not None:
        Y_val = drop_class(filter_model, X_val, Y_val, device,
                           confidence_cutoff)

    expected_accuracy = compute_expected_accuracy(Y_val)

    X_val, Y_val = to_device(X_val, Y_val, device)

    print(f"Expected Accuracy: {expected_accuracy:.4f}")
    print('---' * 10)

    for epoch in range(epochs):
        average_loss = 0
        total_samples = 0
        np.random.shuffle(train_files)
        for i in range(0, len(train_files), buffer_size):
            files = train_files[i:i + buffer_size]

            X, Y = data_loader(files)
            if filter_model is not None:
                Y = drop_class(filter_model, X, Y, device, confidence_cutoff)

            total_samples += len(X)
            for j in range(0, len(X), batch_size):
                X_batch = X[j:j + batch_size]
                Y_batch = Y[j:j + batch_size]

                X_batch, Y_batch = to_device(X_batch, Y_batch, device)

                loss = train_one_epoch(model, X_batch, Y_batch, loss_function,
                                       optimizer)
                average_loss += loss

        average_loss /= total_samples

        val_loss, val_accuracy = validate(model, X_val, Y_val, loss_function)

        kappa = compute_kappa(val_accuracy, expected_accuracy)

        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            save_model(best_model, save_path)

        print(f"Epoch {epoch + 1:03d}/{epochs:03d} - "
              f"Training Loss: {average_loss:.4f}, "
              f"Validation Loss: {val_loss:.4f}, "
              f"Validation Accuracy: {val_accuracy:.4f}, "
              f"Cohen's Kappa: {kappa:.4f}")

    return best_model


def test_model(model, data_directory, model_directory, file_name, device,
               loss_function):

    test_files = glob.glob(os.path.join(data_directory, 'Test/*.pkl'))
    X_test, Y_test = data_loader(test_files)

    expected_accuracy = compute_expected_accuracy(Y_test)

    X_test, Y_test = to_device(X_test, Y_test, device)

    test_loss, test_accuracy = validate(model, X_test, Y_test, loss_function)
    kappa = compute_kappa(test_accuracy, expected_accuracy)

    report = f"Test Loss: {test_loss:.4f}, " \
                f"Test Accuracy: {test_accuracy:.4f}, " \
                f"Test Expected Accuracy: {expected_accuracy:.4f}, " \
                f"Cohen's Kappa: {kappa:.4f}"
    print(report)

    report_file = os.path.join(model_directory, file_name)
    with open(report_file, 'w') as f:
        f.write(report)


if __name__ == '__main__':
    configurations = load_configurations()
    directories = parse_directories()

    data_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['classifier'])
    model_directory = os.path.join(
        BASEPATH, directories['internal']['intermediate']['models'])

    device = configurations['model']['device']
    layer_sizes = configurations['model']['layer_sizes']
    num_classes = configurations['model']['num_classes']

    model = DisciplineClassifier(layer_sizes, num_classes).to(device)
    loss_function = nn.BCELoss()

    pretrain_configurations = configurations['pretraining']
    train_configurations = configurations['training']
    tune_configurations = configurations['finetuning']
    learning_rate = pretrain_configurations['learning_rate']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mono_directory = os.path.join(data_directory, 'Monolabel')
    multi_directory = os.path.join(data_directory, 'Multilabel')

    print("Pretraining the model...")

    model = train_model(model, None, mono_directory, model_directory,
                        'discipline_classification_model_pretrained',
                        pretrain_configurations, device, loss_function,
                        optimizer)
    print("Pretraining completed.")
    print('---' * 10)

    filter_model = deepcopy(model)
    filter_model.eval()
    learning_rate = train_configurations['learning_rate']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    print("Training the model...")
    model = train_model(model, filter_model, multi_directory, model_directory,
                        'discipline_classification_model_trained',
                        train_configurations, device, loss_function, optimizer)

    print("Training completed.")
    print('---' * 10)

    print("Finetuning the model...")

    learning_rate = tune_configurations['learning_rate']

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = train_model(model, None, mono_directory, model_directory,
                        'discipline_classification_model_finetuned',
                        tune_configurations, device, loss_function, optimizer)

    print("Finetuning completed.")
    print('---' * 10)

    test_model(model, multi_directory, model_directory,
               'multi_label_report.txt', device, loss_function)

    test_model(model, mono_directory, model_directory, 'mono_label_report.txt',
               device, loss_function)
