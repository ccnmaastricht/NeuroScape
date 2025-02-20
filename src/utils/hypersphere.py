import numpy as np


def convert_to_polar(x):
    """
    Converts the input data to polar coordinates.

    Parameters:
    x (np.ndarray): The input data.

    Returns:
    phi (np.ndarray): The angle of the input data.
    radius (np.ndarray): The radius of the input data.
    """
    num_observations, num_dimensions = x.shape
    phi = np.zeros((num_observations, num_dimensions - 1))
    squares = x**2
    radius = np.sqrt(squares.sum(axis=1))
    for i in range(num_dimensions - 1):
        phi[:, i] = np.arctan2(np.sqrt(squares[:, i + 1:].sum(axis=1)), x[:,
                                                                          i])

    return phi, radius


def convert_to_cartesian(phi, radius=None):
    """
    Converts the input data to cartesian coordinates.

    Parameters:
    phi (np.ndarray): The angle of the input data.
    radius (np.ndarray): The radius of the input data.

    Returns:
    x (np.ndarray): The input data in cartesian coordinates.
    """
    num_observations, num_dimensions = phi.shape
    x = np.zeros((num_observations, num_dimensions + 1))
    sines = np.sin(phi)
    cosines = np.cos(phi)
    x[:, 0] = cosines[:, 0]
    x[:, -1] = np.prod(sines, axis=1)
    for i in range(1, num_dimensions):
        x[:, i] = cosines[:, i] * np.prod(sines[:, :i], axis=1)

    if radius is not None:
        return x * np.expand_dims(radius, axis=1)
    else:
        return x


def mean_angle(phi):
    """
    Calculates the mean angle of the input data.

    Parameters:
    phi (np.ndarray): The angle of the input data.

    Returns:
    mean_angle (np.ndarray): The mean angle of the input data.
    """
    return np.angle(np.mean(np.exp(1j * phi), axis=0)).__abs__()


def get_centroids(x, labels):
    """
    Calculates the spherical centroids of the input data.

    Parameters:
    x (np.ndarray): The input data.
    labels (np.ndarray): The labels of the input data.

    Returns:
    centroids (np.ndarray): The centroids of the input data.
    """
    unique_labels = np.unique(labels)
    centroids = np.zeros((unique_labels.size, x.shape[1]))
    phi, _ = convert_to_polar(x)
    for i, label in enumerate(unique_labels):
        centroid = mean_angle(phi[labels == label]).reshape(1, -1)
        centroids[i] = convert_to_cartesian(centroid)

    return centroids
