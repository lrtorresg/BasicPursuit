
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

# Function that reshapes a csr_matrix csr, so the shape of the 
# returned csr_matrix is equal to new_shape.
# This function works in similar way that reshape works for octave. 

def reshape(csr, new_shape):
    
    if not hasattr(new_shape, '__len__') or len(new_shape) != 2:
        raise ValueError('`shape` must be a sequence of two integers')
    
    coo = csr.tocoo()

    input_nb_rows, input_nb_cols = coo.shape
    input_size = input_nb_rows * input_nb_cols

    new_size =  new_shape[0] * new_shape[1]
    
    if new_size != input_size:
        raise ValueError('total size of new array must be unchanged')
    
    flat_indices = input_nb_rows * coo.col + coo.row
    new_col_indices, new_row_indices = divmod(flat_indices, new_shape[0])

    coo = coo_matrix((coo.data, (new_row_indices, new_col_indices)), shape = new_shape)
    
    csr = coo.tocsr()
    
    return csr
