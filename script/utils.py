import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import nibabel as nb
from sklearn.preprocessing import StandardScaler


def load_nii(filename, verbose=False):
    '''
    This function returns an array containg data from the file called filename
    Inputs:
        filename
    Output:
        data array
    '''
    if verbose:
        print('Reading the file', filename)
    return nb.load(filename).get_fdata()


def save_nii(filename, data, verbose=False):
    '''
    This function save data in the nifti format
    Inputs:
        filename: is the filename
        data: is the data to be saved
    '''
    if verbose:
        print('Saving the file', filename)
    nb.save(nb.Nifti1Image(data, np.eye(4)), filename)


def signal_as_toeplitz(kernel=[1], signal_length=15):
    kernel = kernel.reshape((-1,))
    return toeplitz(list(kernel) + [0] * (signal_length - 1),
                    [0] * signal_length)[:-len(kernel) + 1]


def var_parameters_2_A(Amat):
    N_nds = Amat.shape[1]
    p = int(Amat.shape[0] / N_nds)
    # print(f'the model order is {p}')
    Arec = []
    for lag in range(p):
        Alag = Amat[lag::p, :].T
        Arec.append( Alag[:, :, np.newaxis] )
    return np.squeeze(np.concatenate(Arec, axis=2))


def force_lower_triangular(A):
    N_nds = A.shape[0]
    A = np.abs( A )
    M = np.zeros( A.shape )
    for j in range( N_nds-1 ):
        for i in range( j+1, N_nds ):
            M[i, j] = A[i, j] + A[j, i]
            # print(f'A[{i}, {j}] + A[{j}, {i}] = {A[i, j]}+{A[j, i]} = {M[i, j]}')
    return np.tril( M )


def lower_triangular_2_vec(M):
    return M[np.tril_indices(M.shape[0],-1)]


def get_dcm_node_signals(X, node_number):
    node_index = node_number-1
    if X.ndim == 3: return np.squeeze(X[:, node_index, :])
    elif X.ndim == 2: return X[:, node_index]


def signal_2_var(X, p=2):
    N = X.shape[0]
    N_nds = X.shape[1]
    Xvar = []
    Yvar = np.zeros( (N-p, N_nds) )
    for col in range(N_nds):
        Yvar[:, col] = X[p:, col]
        Xvar_tmp = np.zeros( (N-p, p) )
        for i in range(p):
            Xvar_tmp[:, i] = X[ p-i-1 : N-i-1, col]
        Xvar.append( Xvar_tmp )        
    Xvar = np.concatenate( Xvar, axis = 1 )
    return Xvar, Yvar


def get_parents_nodes(node, A):
    A = force_lower_triangular( A )
    Aslice_row = A[node-1, :]
    Aslice_col = A[:, node-1]
    nodes_in = set(np.where( Aslice_row != 0 )[0]+1)
    cc = list(nodes_in)
    cc.sort()
    return np.array(cc)

def plot_dcm(X, A, U, timeline, dcm_index=0, show_connections=False):

    Altri = force_lower_triangular(A)
    N_nds = A.shape[0]
    
    if show_connections:
        from mne_connectivity.viz import plot_connectivity_circle
        names = [f'X{i+1}' for i in range(N_nds)]
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'projection': 'polar'})
        fig, ax = plot_connectivity_circle(Altri, names, facecolor='white', textcolor='black',
                                    node_edgecolor='white', colormap='hot_r',
                                    colorbar_pos=(-1,0.5), fig=fig, ax=ax)
        fig.tight_layout()
        # fig.savefig(os.path.join('figs','exp1_dcm_connectivity_10nodes.pdf'))
        fig.savefig(os.path.join('/content/drive/MyDrive/Balloon/figs','exp1_dcm_connectivity_10nodes.pdf'))
        plt.show()

    # plot time courses
    if X.ndim == 2: ymin, ymax = X.min(), X.max()
    elif X.ndim == 3: ymin, ymax = X[:, :, dcm_index].min(), X[:, :, dcm_index].max()

    fig, axs = plt.subplots(figsize=(10, 10), nrows=N_nds, ncols=1)
    for roi_index in range(N_nds):
        roi_number = roi_index+1 
        ax = axs[ roi_index ]

        # signal of roi_number=roi_index+1 and dcm_instance=dcm_index+1          
        if X.ndim == 2: y = X[:, roi_index]
        elif X.ndim == 3: y = X[:, roi_index, dcm_index]

        ax.plot(timeline, y, label=r'$\mathbf{x}_{'+f'{roi_number}'+'}$')
        # show stimulus of the current node in blue
        u = U[:, roi_index]
        pseudoevents = [timeline[i] for i in range( len(u) ) if u[i] == 1]
        for e in pseudoevents: ax.axvline(x=e, ymin=ymin, ymax=ymax, linewidth=2, color='r', alpha=0.5)
        # show stimulus of connected nodes in green
        parents_nodes = get_parents_nodes(roi_number, A)
        if len(parents_nodes) > 0: 
            for parent_node in parents_nodes:
                u = U[:, parent_node-1]
                pseudoevents = [timeline[i] for i in range( len(u) ) if u[i] == 1]
                for e in pseudoevents: ax.axvline(x=e, ymin=ymin, ymax=ymax, linewidth=2, color='g', alpha=0.5)
            parents_str = r' $\leftarrow$' + ', '.join(map(lambda x: r'$\mathbf{x}_{'+f'{x}'+'}$', list(parents_nodes)))
        else: parents_str = ' (no parents)'

        node_str = r'$\mathbf{x}_{'+f'{roi_number}'+ r'}$'
        ax.set_title(node_str + parents_str)
        ax.set_ylim([ymin, ymax])
        # ax.legend()
        if roi_number == N_nds: ax.set_xlabel('time (seconds)')
        else:
            ax.xaxis.set_tick_params(labelbottom=False)
            ax.set_xticks([])
    
    fig.tight_layout()
    fig.savefig(os.path.join('/content/drive/MyDrive/Balloon/figs','exp1_dcm_10nodes.pdf'))
    plt.show()




#################### test functions
def test_signal_2_var():
    # testing signal_2_var()
    N_nds = 3
    N_pts = 6
    p = 3
    # X = np.concatenate([np.arange(1,N_pts+1).reshape((-1, 1)) for _ in range(N_nds)], axis=1)
    X = np.arange(N_nds * N_pts).reshape((-1, N_nds), order='F') + 1
    Xvar, Yvar = signal_2_var(X, p=p)

    # print X
    print('X (each column is one ROI signal)')
    print('\t'.join([f'roi {roi_n+1}' for roi_n in range(N_nds)]))
    for row in range(N_pts):
        print('\t'.join([f'x({x})' for x in X[row, :]]))

    print(f'\nfor p={p}\n')

    # print Yvar, Xvar
    y_hdr = '\t'.join([f'x{roi_n+1}(t)' for roi_n in range(N_nds)])
    x_hdr = '\t'.join([f'x{roi_n+1}(t-{j})' for roi_n in range(N_nds) for j in range(1,p+1)])
    var_hdr = y_hdr +'\t|  '+x_hdr
    print(var_hdr)
    for row in range(Yvar.shape[0]):
        y_row = '\t'.join( [f'y({int(y)})' for y in Yvar[row, :]] )
        x_row = '\t'.join( [f' x({int(x)})' for x in Xvar[row, :]] )
        data_row = y_row +'\t|  '+x_row
        print(data_row)


def test_var_parameters_2_A():
    # testing var_parameters_2_A()
    N_nds = 3
    p = 3
    # X = np.concatenate([np.arange(1,N_pts+1).reshape((-1, 1)) for _ in range(N_nds)], axis=1)
    
    parameters = []
    for col in range(N_nds):
        pcol = []
        for i in range(N_nds):
            for j in range(p):
                pcol.append( f'tar_n{col+1},in_n{i+1},lag{j+1}' ) 
        parameters.append( np.array(pcol).reshape((-1, 1)) )
    parameters = np.concatenate(parameters, axis = 1)
    print(f'\ntest of var_parameters_2_A')
    print(f'\nmatrix of parameters')
    print(parameters)
    print(f'\n')

    # theta = np.arange(N_nds * N_nds * p).reshape((-1, N_nds), order='F') + 1
    # print( theta )
    Arec = var_parameters_2_A(parameters)
    for j in range(p):
        print(f'\nmatrix of lag {j+1}')
        print(Arec[:, :, j])




def test_forceltri_and_ltri2vec():
    # testing force_lower_triangular() and 
    N_nds = 4
    X = np.arange(1,N_nds**2 + 1).reshape((N_nds, N_nds))
    Xltri = force_lower_triangular(X)
    Xltrivec = lower_triangular_2_vec(Xltri)

    print(f'\nA')
    print(X)
    print(f'\nA-lower-triangular')
    print(Xltri)
    print(f'\nA-lower-triangular as vector')
    print(Xltrivec)



if __name__ == '__main__':
    test_signal_2_var()
    # test_var_parameters_2_A()
    # test_forceltri_and_ltri2vec()
