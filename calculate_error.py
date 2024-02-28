import numpy as np


def calculate_q_t_error(q_true: np.ndarray, t_true: np.ndarray, q_estimated: np.ndarray, t_estimated: np.ndarray,
                        verbose: bool=False) -> tuple[float, float]:
    
    assert isinstance(q_true, np.ndarray)
    assert isinstance(t_true, np.ndarray)
    assert isinstance(q_estimated, np.ndarray)
    assert isinstance(t_estimated, np.ndarray)
    
    t_estimated = t_estimated.flatten()
    t_true = t_true.flatten()
    
    assert len(q_true.shape) == 1 and q_true.shape[0] == 4
    assert len(t_true.shape) == 1 and t_true.shape[0] == 3
    assert len(q_estimated.shape) == 1 and q_estimated.shape[0] == 4
    assert len(t_estimated.shape) == 1 and t_estimated.shape[0] == 3
    
    if verbose:
        print(f'{q_true=}')
        print(f'{t_true=}')
        print(f'{q_estimated=}')
        print(f'{t_estimated=}')
    
    eps = 1e-15

    q_estimated = q_estimated / (np.linalg.norm(q_estimated) + eps)
    q_true = q_true / (np.linalg.norm(q_true) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q_estimated * q_true)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    t_estimated = t_estimated / (np.linalg.norm(t_estimated) + eps)
    t_true = t_true / (np.linalg.norm(t_true) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t_estimated * t_true)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    assert not (np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t))), 'This should never happen! Debug here'

    return err_q, err_t


# def calculate_AUC(err_qt, degree1=5, degree2=20):
#     assert degree1 < degree2
#     if len(err_qt) > 0:
#         err_qt = np.asarray(err_qt)
#         # Take the maximum among q and t errors
#         err_qt = np.max(err_qt, axis=1)
#         # Convert to degree
#         err_qt = err_qt * 180.0 / np.pi
#         # Make infs to a large value so that np.histogram can be used.
#         err_qt[err_qt == np.inf] = 1e6
#
#         # Create histogram
#         bars = np.arange(degree2 + 1)
#         qt_hist, _ = np.histogram(err_qt, bars)
#         # Normalize histogram with all possible pairs
#         num_pair = float(len(err_qt))
#         qt_hist = qt_hist.astype(float) / num_pair
#         # Make cumulative
#         qt_acc = np.cumsum(qt_hist)
#     else:
#         qt_acc = [0] * degree2
#        
#     return np.mean(qt_acc[:degree1]), np.mean(qt_acc)


def calculate_error(img1_no: int, img2_no: int, dataset: str, method: str) -> tuple[float, float]:
    q_true = np.loadtxt(f'poses/{dataset}/q-{img1_no}-{img2_no}.txt')
    t_true = np.loadtxt(f'poses/{dataset}/t-{img1_no}-{img2_no}.txt')
    q_estimated = np.loadtxt(f'poses/{dataset}/{method}/q-{img1_no}-{img2_no}_estimated.txt')
    t_estimated = np.loadtxt(f'poses/{dataset}/{method}/t-{img1_no}-{img2_no}_estimated.txt')

    return calculate_q_t_error(q_true, t_true, q_estimated, t_estimated)
