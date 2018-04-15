import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    Hk_half, Wk_half = int(Hk/2), int(Wk/2)
    out = np.zeros((Hi, Wi))
    # print(image)
    ### YOUR CODE HERE
    for i in range(Hi):
        for j in range(Wi):
            for m in range(Hk):
                for n in range(Wk):
                    y = i - Hk_half + m
                    x = j - Wk_half + n
                    # print((i, j), (y, x), (m, n))
                    if x >= 0 and y >= 0 and x < Wi and y < Hi:
                        out[i, j] = out[i, j] + image[y, x] * kernel[m, n]
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W)
        pad_width: width of the zero padding (left and right padding)
        pad_height: height of the zero padding (bottom and top padding)

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width)
    """

    ### YOUR CODE HERE
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant')
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    Hk_half, Wk_half = int(Hk/2), int(Wk/2)
    out = np.zeros((Hi, Wi))
    ### YOUR CODE HERE
    image = zero_pad(image, Hk_half, Wk_half)
    # kernel = np.flip(np.flip(kernel, 1), 0)
    for i in range(Hi):
        for j in range(Wi):
            out[i, j] = \
                np.sum(kernel * image[i:i + Hk, j:j + Wk])
    ### END YOUR CODE

    return out

def conv_faster(image, kernel):
    """
    Args:
        image: numpy array of shape (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk)

    Returns:
        out: numpy array of shape (Hi, Wi)
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of f and g

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    out = conv_fast(f, np.flip(np.flip(g, 0), 1))
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of f and g

    Subtract the mean of g from g so that its mean becomes zero

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """

    out = None
    ### YOUR CODE HERE
    ave = np.sum(g) / (g.shape[0] * g.shape[1])
    out = cross_correlation(f, g - ave)
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of f and g

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Args:
        f: numpy array of shape (Hf, Wf)
        g: numpy array of shape (Hg, Wg)

    Returns:
        out: numpy array of shape (Hf, Wf)
    """
    ### YOUR CODE HERE
    Hi, Wi = f.shape
    Hk, Wk = g.shape
    # print(f.shape, g.shape)
    g_mean = np.sum(g)/ (Hk*Wk)
    g_standard = np.sum(np.abs(g - g_mean)) / (Hk*Wk)
    kernel = (g - g_mean) / g_standard

    out = np.zeros((Hi, Wi))
    image = zero_pad(f, int(Hk/2), int(Wk/2))
    # print(image.shape, kernel.shape)
    for i in range(Hi):
        for j in range(Wi):
            f_mean = np.sum(image[i:i + Hk, j:j + Wk]) / (Hk * Wk)

            f_standard = np.sum(np.abs(image[i:i + Hk, j:j + Wk] - f_mean)) / (Hk * Wk)

            out[i, j] = np.sum(kernel * (image[i:i + Hk, j:j + Wk] - f_mean) / f_standard)
    ### END YOUR CODE

    return out
