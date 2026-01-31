import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    # Write code here
    first = np.dot(x_t, Wx)
    second = np.dot(h_prev, Wh)
    calc = first + second + b
    h_t = np.tanh(calc)
    return h_t