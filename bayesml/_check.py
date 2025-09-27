# Code Author
# Yuta Nakahara <y.nakahara@waseda.jp>
# Yuji Iikubo <yuji-iikubo.8@fuji.waseda.jp>
# Yasushi Esaki <esakiful@gmail.com>
# Jun Nishikawa <jun.b.nishikawa@gmail.com>
import numpy as np

_EPSILON = np.sqrt(np.finfo(np.float64).eps)

def float_in_closed01(val,val_name,exception_class):
    if np.issubdtype(type(val),np.floating):
        if val >= 0.0 and val <= 1.0:
            return val
    if np.issubdtype(type(val),np.integer):
        if val >= 0.0 and val <= 1.0:
            return float(val)
    raise(exception_class(val_name + " must be in [0,1]."))

def pos_float(val,val_name,exception_class):
    if np.issubdtype(type(val),np.floating):
        if val > 0.0:
            return val
    if np.issubdtype(type(val),np.integer):
        if val > 0.0:
            return float(val)
    raise(exception_class(val_name + " must be positive (not including 0.0)."))

def pos_int(val,val_name,exception_class):
    if np.issubdtype(type(val),np.integer):
        if val > 0:
            return val
    raise(exception_class(val_name + " must be int. Its value must be positive (not including 0)."))

def nonneg_int(val,val_name,exception_class):
    if np.issubdtype(type(val),np.integer):
        if val >= 0:
            return val
    raise(exception_class(val_name + " must be int. Its value must be non-negative (including 0)."))

def nonneg_ints(val,val_name,exception_class):
    try:
        return nonneg_int(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and np.all(val>=0):
            return val
    raise(exception_class(val_name + " must be int or a numpy.ndarray whose dtype is int. Its values must be non-negative (including 0)."))

def pos_ints(val,val_name,exception_class):
    try:
        return pos_int(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and np.all(val>0):
            return val
    raise(exception_class(val_name + " must be int or a numpy.ndarray whose dtype is int. Its values must be positive (not including 0)."))

def int_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim == 1:
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray whose dtype is int."))

def nonneg_int_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim == 1 and np.all(val>=0):
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray whose dtype is int. Its values must be non-negative (including 0)."))

def pos_int_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim == 1 and np.all(val>0):
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray whose dtype is int. Its values must be positive (not including 0)."))

def nonneg_int_vecs(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim >= 1 and np.all(val>=0):
            return val
    raise(exception_class(val_name + " must be a numpy.ndarray whose ndim >= 1 and dtype is int. Its values must be non-negative (including 0)."))


def nonneg_float_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.floating) and val.ndim == 1 and np.all(val>=0):
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray whose dtype is float. Its values must be non-negative (including 0)."))

def int_of_01(val,val_name,exception_class):
    if np.issubdtype(type(val),np.integer):
        if val == 0 or val ==1:
            return val
    raise(exception_class(val_name + " must be int. Its value must be 0 or 1."))

def ints_of_01(val,val_name,exception_class):
    try:
        return int_of_01(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and np.all(val >= 0) and np.all(val <= 1):
            return val
    raise(exception_class(val_name + " must be int or a numpy.ndarray whose dtype is int. Its values must be 0 or 1."))

def int_vec_of_01(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim == 1 and np.all(val >= 0) and np.all(val <= 1):
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray whose dtype is int. Its values must be 0 or 1."))

def scalar(val,val_name,exception_class):
    if np.issubdtype(type(val),np.integer) or np.issubdtype(type(val),np.floating):
        return val
    raise(exception_class(val_name + " must be a scalar."))

def pos_scalar(val,val_name,exception_class):
    if np.issubdtype(type(val),np.integer) or np.issubdtype(type(val),np.floating):
        if val > 0.0:
            return val
    raise(exception_class(val_name + " must be a positive scalar."))

def sym_mat(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if val.ndim == 2 and val.shape[0] == val.shape[1]:
            if np.allclose(val, val.T):
                return val
    raise(exception_class(val_name + " must be a symmetric 2-dimensional numpy.ndarray."))

def pos_def_sym_mat(val,val_name,exception_class):
    sym_mat(val,val_name,exception_class)
    try:
        np.linalg.cholesky(val)
        return val
    except np.linalg.LinAlgError:
        pass
    raise(exception_class(val_name + " must be a positive definite symmetric 2-dimensional numpy.ndarray."))

def sym_mats(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if val.ndim >= 2 and val.shape[-1] == val.shape[-2]:
            if np.allclose(val, np.swapaxes(val,-1,-2)):
                return val
    raise(exception_class(val_name + " must be a symmetric 2-dimensional numpy.ndarray."))

def pos_def_sym_mats(val,val_name,exception_class):
    sym_mats(val,val_name,exception_class)
    try:
        np.linalg.cholesky(val)
        return val
    except np.linalg.LinAlgError:
        pass
    raise(exception_class(val_name + " must be a positive definite symmetric 2-dimensional numpy.ndarray."))

def float_(val,val_name,exception_class):
    if np.issubdtype(type(val),np.floating):
        return val
    if np.issubdtype(type(val),np.integer):
        return float(val)
    raise(exception_class(val_name + " must be a scalar."))

def floats(val,val_name,exception_class):
    try:
        return float_(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer):
            return val.astype(float)
        if np.issubdtype(val.dtype,np.floating):
            return val
    raise(exception_class(val_name + " must be float or a numpy.ndarray."))

def pos_floats(val,val_name,exception_class):
    try:
        return pos_float(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and np.all(val>0):
            return val.astype(float)
        if np.issubdtype(val.dtype,np.floating) and np.all(val>0.0):
            return val
    raise(exception_class(val_name + " must be float or a numpy.ndarray. Its values must be positive (not including 0)"))

def float_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim == 1:
            return val.astype(float)
        if np.issubdtype(val.dtype,np.floating) and val.ndim == 1:
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray."))

def pos_float_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim == 1 and np.all(val>0):
            return val.astype(float)
        if np.issubdtype(val.dtype,np.floating) and val.ndim == 1 and np.all(val>0.0):
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray. Its values must be positive (not including 0)"))
    
def float_vecs(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim >= 1:
            return val.astype(float)
        if np.issubdtype(val.dtype,np.floating) and val.ndim >= 1:
            return val
    raise(exception_class(val_name + " must be a numpy.ndarray whose ndim >= 1."))

def pos_float_vecs(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim >= 1 and np.all(val>0):
            return val.astype(float)
        if np.issubdtype(val.dtype,np.floating) and val.ndim >= 1 and np.all(val>0.0):
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray. Its values must be positive (not including 0)"))

def float_vec_sum_1(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim == 1 and abs(val.sum() - 1.) <= _EPSILON:
            return val.astype(float)
        if np.issubdtype(val.dtype,np.floating) and val.ndim == 1 and abs(val.sum() - 1.) <= _EPSILON:
            return val
    raise(exception_class(val_name + " must be a 1-dimensional numpy.ndarray, and the sum of its elements must equal to 1."))

def float_vecs_sum_1(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim >= 1 and np.all(np.abs(np.sum(val, axis=-1) - 1.) <= _EPSILON):
            return val.astype(float)
        if np.issubdtype(val.dtype,np.floating) and val.ndim >= 1 and np.all(np.abs(np.sum(val, axis=-1) - 1.) <= _EPSILON):
            return val
    raise(exception_class(val_name + " must be a numpy.ndarray whose ndim >= 1, and the sum along the last dimension must equal to 1."))

def int_(val,val_name,exception_class):   
    if np.issubdtype(type(val),np.integer):
        return val
    raise(exception_class(val_name + " must be an integer."))

def ints(val,val_name,exception_class):
    try:
        return int_(val,val_name,exception_class)
    except:
        pass
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer):
            return val
    raise(exception_class(val_name + " must be int or a numpy.ndarray whose dtype is int."))

def onehot_vec(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim == 1 and np.all(val >= 0) and val.sum()==1:
            return val
    raise(exception_class(val_name + " must be a one-hot vector (1-dimensional ndarray) whose dtype must be int."))

def onehot_vecs(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim >= 1 and np.all(val >= 0) and np.all(val.sum(axis=-1)==1):
            return val
    raise(exception_class(val_name + " must be a numpy.ndarray whose dtype is int and whose last axis constitutes one-hot vectors."))

def int_vecs(val,val_name,exception_class):
    if type(val) is np.ndarray:
        if np.issubdtype(val.dtype,np.integer) and val.ndim >= 1:
            return val
    raise(exception_class(val_name + " must be a numpy.ndarray whose dtype is int and ndim >= 1."))
    
def shape_consistency(val: int, val_name: str, correct: int, correct_name: str, exception_class):
    if val != correct:
        message = (f"{val_name} must coincide with {correct_name}: "
                   + f"{val_name} = {val}, {correct_name} = {correct}")
        raise(exception_class(message))
