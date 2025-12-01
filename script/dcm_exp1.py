import numpy as np
import random
import scipy.io
import os
from sklearn.preprocessing import StandardScaler
from utils import save_nii, signal_as_toeplitz
from hrf_spm import spm_hrf_compat 

# def force_lower_triangular(A):
#     N_nds = A.shape[0]
#     A = np.abs( A )
#     M = np.zeros( A.shape )
#     for j in range( N_nds-1 ):
#         for i in range( j+1, N_nds ):
#             M[i, j] = A[i, j] + A[j, i]
#             # print(f'A[{i}, {j}] + A[{j}, {i}] = {A[i, j]}+{A[j, i]} = {M[i, j]}')
#     return np.tril( M )


def split_A_disjoint(A, p_inst=0.5, seed=42):
    """
    Divide A en dos matrices disjuntas:
    - algunas conexiones van SOLO a C_inst
    - otras conexiones van SOLO a B_lag

    p_inst: probabilidad de que una conexión vaya a la parte instantánea.
    """
    rng = np.random.default_rng(seed)
    A = np.array(A, dtype=float)

    C_inst = np.zeros_like(A)
    B_lag  = np.zeros_like(A)

    rows, cols = np.nonzero(A)
    for i, j in zip(rows, cols):
        if rng.random() < p_inst:
            C_inst[i, j] = A[i, j]   # solo instantánea
        else:
            B_lag[i, j]  = A[i, j]   # solo lag

    return C_inst, B_lag


def get_random_A(N_nds=10, N_conn=10):
    A = -np.eye(N_nds)
    node_pairs = []
    for i in range(N_nds):
        for j in range(N_nds):
            if i > j:
                node_pairs.append( (i,j) ) 
    nodes_connected = []
    while len(nodes_connected) < N_conn:
        n = random.choice( node_pairs )
        if n not in nodes_connected: nodes_connected.append( n )
    for i, j in nodes_connected:
        A[i, j] = random.choice(np.arange(0.5,0.9,0.1))
    print(A)
    return A


def get_node_events(timeline, nblocks=5, duration_range=(1,3)): # u for one node
    N_pts = len(timeline)
    time_offset = 10
    possible_onsets = np.arange(time_offset, N_pts-time_offset)
    sep_bwn_ons = 10
    durations = []
    onsets = []    
    for i in range(nblocks):
        current_block_duration = random.randint(duration_range[0], duration_range[1])
        current_onset = random.choice( possible_onsets )
        onsets.append( current_onset )
        durations.append( current_block_duration )
        possible_onsets = possible_onsets[np.logical_or( possible_onsets < current_onset - sep_bwn_ons,
                                                possible_onsets > current_onset + sep_bwn_ons)]
    u = np.zeros(N_pts)
    for i,d in zip(onsets, durations):
        u[i:i+d] = 1
    return u, timeline, onsets, durations


def get_network_events(timeline, N_nds, nblocks=5, duration_range=(1,3)):
    N_pts = len(timeline)
    U = np.zeros(shape=(N_pts,N_nds))
    for n in range( N_nds ):
        U[:,n],_,_,_ = get_node_events(timeline, nblocks, duration_range)
    return U


def get_noise_matrix(timeline, N_signals = 1000):
    TR_noise, N_pts_noise = 1, 300
    timeline_noise = np.arange(0, TR_noise*N_pts_noise, TR_noise)
    noise = scipy.io.loadmat(os.path.join('/content/drive/MyDrive/Balloon/scripts', 'noise_nodrift.mat'))  ######################
    #noise = scipy.io.loadmat(os.path.join('scripts', 'noise_nodrift.mat'))
    N_noise, N_pts = noise['noise'].shape
    noise_signals = np.zeros( (len(timeline), N_signals) )
    for i in range(N_signals):
        j = np.random.randint(0, N_noise)
        noise_signal = noise['noise'][j,:] 
        noise_signal = np.interp(timeline, timeline_noise, noise_signal)
        noise_signal = noise_signal - np.mean(noise_signal)
        noise_signals[:, i] = noise_signal / np.std(noise_signal)
    return noise_signals


def add_noise_dcm(X, timeline, SNR):
    noise_signals = np.zeros(X.shape)
    for k in range(X.shape[2]):
        noise_signals[:, :, k] = get_noise_matrix( timeline, N_signals=X.shape[1] )    
    if SNR == 0:
        w = 0
    else:        
        w = 1/SNR #X.max() / SNR
    return X + w * noise_signals


def add_noise_dcm_gsn(X, SNR):
    N_pts, N_nds, N_instances = X.shape
    if SNR == 0:
        w = 0
    else:        
        w = 1/SNR #X.max() / SNR
    return X + w * np.random.randn(N_pts, N_nds, N_instances)


def get_dcm_instance( timeline, U, A ):
    N_pts, N_nds = U.shape
    X = np.zeros((N_pts, N_nds))
    X_bold = np.zeros((N_pts, N_nds))
    dt = np.ones(N_nds)
    for t in range(1,N_pts):
        X[t,:] = X[t-1,:] + dt * ( np.dot(A, X[t-1, :]) + U[t, :] )
    h = spm_hrf_compat(timeline)
    H = signal_as_toeplitz(h, N_pts)
    for n in range(N_nds):         
        s = np.dot(H, X[:,n])
        s = s - s.mean()
        X_bold[:,n] = s / s.std()
    return X_bold, X

from sklearn.preprocessing import StandardScaler
from utils import signal_as_toeplitz
from hrf_spm import spm_hrf_compat

def stabilize_C_B(C_inst, B_lag, max_radius=0.8):
    """
    Reescala C_inst y B_lag para que la dinámica sea estable.

    max_radius: radio espectral máximo deseado de la matriz efectiva.
    """
    # Matriz efectiva aproximada (solo para medir la escala)
    A_eff = C_inst + B_lag

    eigvals = np.linalg.eigvals(A_eff)
    rad = np.max(np.abs(eigvals))

    if rad > max_radius and rad > 0:
        scale = max_radius / rad
        C_inst = C_inst * scale
        B_lag  = B_lag * scale

    return C_inst, B_lag


def get_dcm_instance_svar1(timeline, U, A_custom, p_inst=0.5, seed=42):
    """
    Simulación neuronal con modelo SVAR(1) + HRF.

    (I - C_inst) X_t = B_lag X_{t-1} + U_t

    p_inst: probabilidad de que una conexión vaya a la parte instantánea.
    """
    U = np.asarray(U, dtype=float)
    N_pts, N_nds = U.shape

    # 1) Separación disjunta de la conectividad
    C_inst, B_lag = split_A_disjoint(A_custom, p_inst=p_inst, seed=seed)

    # 2) Estabilizar la dinámica (evitar explosión numérica)
    C_inst, B_lag = stabilize_C_B(C_inst, B_lag, max_radius=0.8)

    # 3) Dinámica neuronal SVAR(1)
    X = np.zeros((N_pts, N_nds))
    I = np.eye(N_nds)
    M = I - C_inst  # matriz del lado izquierdo; con rad<=0.8 debe ser bien condicionada

    for t in range(1, N_pts):
        rhs = B_lag @ X[t-1, :] + U[t, :]
        X[t, :] = np.linalg.solve(M, rhs)

    # 4) HRF y convolución
    h = spm_hrf_compat(timeline)
    H = signal_as_toeplitz(h, N_pts)

    X_bold = np.zeros_like(X)
    for n in range(N_nds):
        s = H @ X[:, n]
        s = s - s.mean()
        s = s / s.std()
        X_bold[:, n] = s

    # 5) Estandarización
    X_bold = StandardScaler().fit_transform(X_bold)
    X = StandardScaler().fit_transform(X)

    return X_bold, X, C_inst, B_lag



def write_dcm_dataset_exp1(
    timeline,
    N_nds=10,
    N_conn=10,
    N_subjects=100,
    N_instances=100,
    nblocks=5,
    duration_range=(1,3),
    SNRs=[0],
    data_dirname='data_dcm_exp1',
    noise_type='invivo',
    A_custom=None     #########################################
):
    N_pts = len(timeline)

    # Si me pasan una A_custom, fuerzo N_nds y (opcionalmente) N_conn a ser consistentes
    if A_custom is not None:
        A_custom = np.array(A_custom, dtype=float)
        N_nds = A_custom.shape[0]
        # número de conexiones off-diagonal (solo informativo / para nombre de carpeta)
        N_conn = int(np.sum(np.abs(A_custom) > 0) - N_nds)

    if not os.path.exists(data_dirname):
        os.mkdir(data_dirname)

    dirdataset = os.path.join(data_dirname, f'nds{N_nds}_conn{N_conn}')
    if not os.path.exists(dirdataset):
        os.mkdir(dirdataset)

    for s in range(N_subjects):
    
        dirsubject = os.path.join(f'{dirdataset}', f'sub{s+1}')
        if not os.path.exists(dirsubject):
            os.mkdir(dirsubject)

        Xs = np.zeros((N_pts, N_nds, N_instances))
        Xs_bold = np.zeros((N_pts, N_nds, N_instances))

        # Entradas de la red (pseudoeventos)
        U = get_network_events(timeline, N_nds, nblocks, duration_range)

        # Aquí elegimos la matriz A
        if A_custom is not None:
            A = A_custom.copy()
        else:
            A = get_random_A(N_nds=N_nds, N_conn=N_conn)

        # Simulación DCM para cada instancia
        for i in range(N_instances):
            Xs_bold[:, :, i], Xs[:, :, i] = get_dcm_instance(timeline, U, A)
            Xs_bold[:, :, i] = StandardScaler().fit_transform(Xs_bold[:, :, i])
            Xs[:, :, i] = StandardScaler().fit_transform(Xs[:, :, i])
    
        # Guardar ground truth y entradas
        save_nii(os.path.join(dirsubject, 'A.nii'), A, verbose=True)
        save_nii(os.path.join(dirsubject, 'U.nii'), U, verbose=True)
 
        # Guardar BOLD con diferentes SNR
        for snr in SNRs:
            if noise_type == 'invivo':
                Xs_bold_noisy = add_noise_dcm(Xs_bold, timeline, snr)
            else:
                Xs_bold_noisy = add_noise_dcm_gsn(Xs_bold, snr)
            save_nii(os.path.join(dirsubject, f'X_snr{snr}.nii'), Xs_bold_noisy, verbose=True)


# def write_dcm_dataset_exp1(timeline, N_nds=10, N_conn=10, N_subjects=100, N_instances=100, nblocks=5, duration_range=(1,3), SNRs=[0], data_dirname='data_dcm_exp1', noise_type='invivo'):
#     N_pts = len(timeline)
#     if not os.path.exists(data_dirname): os.mkdir(data_dirname)
#     dirdataset = os.path.join(data_dirname, f'nds{N_nds}_conn{N_conn}')
#     if not os.path.exists(dirdataset): os.mkdir(dirdataset)

#     for s in range(N_subjects):
    
#         dirsubject = os.path.join(f'{dirdataset}', f'sub{s+1}')        
#         if not os.path.exists(dirsubject): os.mkdir(dirsubject)

#         Xs = np.zeros( (N_pts, N_nds, N_instances) )
#         Xs_bold = np.zeros( (N_pts, N_nds, N_instances) )
#         U = get_network_events(timeline, N_nds, nblocks, duration_range)
#         A = get_random_A(N_nds=N_nds, N_conn=N_conn)

#         for i in range(N_instances):
#             Xs_bold[:,:,i], Xs[:,:,i] = get_dcm_instance(timeline, U, A)
#             Xs_bold[:,:,i] = StandardScaler().fit_transform(Xs_bold[:,:,i])
#             Xs[:,:,i] = StandardScaler().fit_transform(Xs[:,:,i])
    
#         save_nii(os.path.join(dirsubject, 'A.nii'), A, verbose=True)
#         save_nii(os.path.join(dirsubject, 'U.nii'), U, verbose=True)
 
#         for snr in SNRs:
#             if noise_type == 'invivo': Xs_bold_noisy = add_noise_dcm(Xs_bold, timeline, snr)
#             else: Xs_bold_noisy = add_noise_dcm_gsn(Xs_bold, snr)
#             save_nii(os.path.join(dirsubject, f'X_snr{snr}.nii'), Xs_bold_noisy, verbose=True)