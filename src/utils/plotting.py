def convert_size(width, height):
    """
    Convert figure size from mm to inches.

    Parameters
    ----------
    width : float
        The width in mm.
    height : float
        The height in mm.

    Returns
    -------
    tuple
        The width and height in inches.
    """

    width = width / 25.4
    height = height / 25.4

    return (width, height)
