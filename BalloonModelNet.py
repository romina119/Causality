# adapted from https://github.com/rmillin/balloon-model/
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import ode, solve_ivp
import random as rn
import os

def split_A_disjoint(A, p_inst=0.5, seed=42):   ############ nueva
    """
    Divide A en dos matrices disjuntas:
    - algunas conexiones van SOLO a C_inst (instantáneo)
    - otras conexiones van SOLO a B_lag (con lag)

    p_inst: probabilidad de que una conexión vaya a la parte instantánea.
    """
    import numpy as np
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


def add_lag_to_neural(neur, B_lag, scale=1.0):    ####### nueva
    """
    Aplica un sistema VAR(1) estable sobre la señal neuronal:
    
        X_t = neur_t + scale * (B_lag @ X_{t-1})

    Donde B_lag es estabilizado automáticamente.
    """
    import numpy as np

    T, N = neur.shape
    neur = np.asarray(neur, dtype=float)
    X = np.zeros_like(neur)

    # ---- estabilización ----
    # reducir magnitud si es necesario
    eigvals = np.linalg.eigvals(B_lag)
    rad = np.max(np.abs(eigvals))
    if rad >= 0.99:
        B_lag = B_lag * (0.99 / rad)

    # ---- dinámica tipo VAR estable ----
    X[0] = neur[0]
    for t in range(1, T):
        X[t] = neur[t] + scale * (B_lag @ X[t-1])

    return X


def neureq(t, I, params): # Eq14 Buxton
    kappa1, tau, N0, interpolators, A, C = params
    ut = np.array([f(t) for f in interpolators])
    N = ut - C @ I
    # N = ut - I
    if np.any(N > -N0):
        dIdt = (kappa1/tau) * N - A@I
    else:
        dIdt = (-kappa1/tau) * N0 - A@I # neural response cannot go below 0
    return dIdt


# # make the neural model function
def StimulusToNeural(timing, u, const, C):
    
    Nsamples, Nrois = u.shape
    # default parameter values
    kappa1 = 2 # (0-3)
    tau = 2 # (1-3)
    N0 = 0
    newdeltat = 1

    newtiming = np.arange(min(timing), max(timing), newdeltat) # higher sampling

    if u.ndim == 1: u = u[:, np.newaxis]

    newu_list = list()
    for j in range(u.shape[1]):
        f_interp = interp1d(timing, u[:, j], kind='nearest')
        newu = f_interp(newtiming)
        newu_list.append(newu[:, np.newaxis] * const)
    newu = np.concatenate( newu_list, axis=1 )

    interpolators = [interp1d(newtiming, newu[:, j], kind='nearest') for j in range(u.shape[1])]

    A = (1/tau) * np.eye(Nrois)
    I0 = - N0

    # Make time array for solution
    t = newtiming
    # Bundle parameters for ODE solver
    params = [kappa1, tau, N0, interpolators, A, C]
    # Call the ODE solver
    #sol = solve_ivp(neureq, [min(t), max(t)], Nrois*[I0], args=(params,))
    sol = solve_ivp(neureq ,[min(t), max(t)], Nrois*[I0], args=(params,), t_eval=newtiming, max_step=np.diff(newtiming).min()/4)   ########################################################

    I = sol.y.T
    newstim = np.concatenate([f(sol.t)[:, np.newaxis] for f in interpolators], axis=1)   ##################################################################
    neur = newstim - I
    neur[neur<0] = 0 # imposes that N0+N>=0
    neur = np.clip(neur, 0, 3.0)   ################################################################
    
    return sol.t, neur

# make the function that converts neural response to flow in the vasculature

def floweq(t, y, params):
    x1, x2 = y
    kappa2, gamma, neuronalact, timing = params
    neuronal = np.interp(t, timing, neuronalact)
    derivs = [x2, neuronal-x2*kappa2-(x1-1)*gamma]
    return derivs


def NeuraltoFlow(timing, neuralresp):
    
    kappa2 = .65; # prior from the paper: 0.65
    gamma = .41; # prior from the paper: 0.41

    # Bundle parameters for ODE solver
    params = [kappa2, gamma, neuralresp, timing]

    # Bundle initial conditions for ODE solver
    f0 = 1. # flow in to vasculature
    s0 = 0. # signal to the vasulature
    y0 = [f0, s0]

    # Make time array for solution
    tInc = 1
    t = np.arange(timing[0], timing[-1], tInc)

    # Call the ODE solver
    # psoln = odeint(systemeq, y0, t, args=(params,))
    solver = ode(floweq).set_integrator('dopri5')
    # solver = ode(floweq).set_integrator("dop853")
    solver.set_initial_value(y0).set_f_params(params)

    k = 0
    soln = [y0]
    
    while solver.successful() and solver.t < t[-1]:
        k += 1
        solver.integrate(t[k])
        soln.append(solver.y)

    # Convert the list to a numpy array.
    psoln = np.array(soln)
        
    flow = psoln[:,0]
    vascsignal = psoln[:,1]
    flow = np.clip(flow,1e-6, 8.0) #prevents inestabilities

    return t, flow, vascsignal

# equation for oxygen extraction

def Eeq(E0, flowin):
    return 1-(1-E0)**(1/flowin)

# balloon model parameter function


def balloonparameter(TE,B0,E0,V0):

    if B0==3:
        r0 = 108
    elif B0==1.5:
        r0 = 15
    else:
        print("""Parameter value for r0 isn't available for the field strength specified, using approximation""")
        r0 = 25 *(B0/1.5)**2 # not sure where Pinglei got this from, but seems approximately correct

    if (B0==3 or B0==1.5):
        epsilon = 0.13 # assuming dominance of macrovascular component
    else:
        raise ValueError('Parameter value for epsilon is not available for the field strength specified')

    v = 40.3 * (B0/1.5)

    k1 = 4.3 * v * E0 * TE
    k2 = epsilon*r0*E0*TE
    k3 = 1 - epsilon

    return (k1, k2, k3, V0, E0)


def systemeq(t, y, params): # system of differential equations for balloon model
    x1, x2 = y
    tau1, tau2, alpha, E0, flowin, timing = params
    f = interp1d(timing, flowin)
    fin = f(t)
    E = Eeq(E0,fin)
    derivs = [(1/tau1)*(fin*E/E0-x1/x2*(x2**(1/alpha)+tau2/(tau1+tau2)*(fin-x2**(1/alpha)))),
             1/(tau1+tau2)*(fin-(x2**(1/alpha)))]
    return derivs

# make the balloon model function

def BalloonModel(timing, flowin, TE):
    
  
    alpha = .4; # 0.32 in Friston
    E0 = 0.4;
    V0 = 0.03; # 0.03 in Buxton, Uludag, et al.
    F0 = 0.01;
    tau2 = 30; # typical value based on fits from Mildner, Norris, Schwarzbauer, and Wiggins (2001)
    B0 = 3; # we have a 3 T scanner!    
    tau1 = V0/F0;
    k1, k2, k3, V0, E0 = balloonparameter(TE, B0, E0, V0);
    q0 = 1;
    v0 = 1;
    V0 = V0;

    # Bundle parameters for ODE solver
    params = [tau1, tau2, alpha, E0, flowin, timing]

    # Bundle initial conditions for ODE solver
    y0 = [q0, v0]

    # Make time array for solution
    tInc = 2.0 #0.1 ################################## produce sobremuestreo ###### mismo que TR
    t = np.arange(timing[0], timing[-1], tInc)

    # Call the ODE solver
    # psoln = odeint(systemeq, y0, t, args=(params,))
    solver = ode(systemeq).set_integrator("dop853")
    solver.set_initial_value(y0).set_f_params(params)

    k = 0
    soln = [y0]
    while solver.successful() and solver.t < t[-1]:
        k += 1
        solver.integrate(t[k])
        soln.append(solver.y)

    # Convert the list to a numpy array.
    psoln = np.array(soln)
        
    q = psoln[:,0]
    v = psoln[:,1]
    bold = V0*(k1*(1-q)+k2*(1-q/v)+k3*(1-v))


    return (t, bold, q, v)
    

# def BalloonNetwork_roiwise(timing, unodes, C):
#     u = np.transpose(C @ unodes.T)
#     deltat = 1
#     const = 1  

#     BalloonNetwork = {'timing': timing, 'u': unodes,
#                       'tneur':[], 'neur':[], 'tflowin':[], 'flowin':[], 
#                       'vascsignal':[], 'tbold': [], 'bold':[], 'v':[], 'q':[]}

#     tneur, neur = StimulusToNeural(timing, u, const, C) # stimulus to neural response
#     BalloonNetwork['tneur'] = tneur
#     BalloonNetwork['neur'] = neur
#     BalloonNetwork['tu'] = timing
#     BalloonNetwork['u'] = unodes
#     BalloonNetwork['Adjacency'] = C
    
#     TE = 0.03
#     Nsamples, Nrois = neur.shape

#     for inode in range(Nrois):
#         print(f'node {inode+1}')
#         tflowin, flowin, vascsignal = NeuraltoFlow(tneur, neur[:, inode]) # neural to flow in
#         tbold, bold, q, v = BalloonModel(tflowin, flowin, TE) # flow in to BOLD
#         BalloonNetwork['tflowin'].append(tflowin[:,np.newaxis])
#         BalloonNetwork['flowin'].append(flowin[:,np.newaxis])
#         BalloonNetwork['vascsignal'].append(vascsignal[:,np.newaxis])
#         BalloonNetwork['tbold'].append(tbold[:,np.newaxis])
#         BalloonNetwork['bold'].append(bold[:,np.newaxis])
#         BalloonNetwork['v'].append(v[:,np.newaxis])
#         BalloonNetwork['q'].append(q[:,np.newaxis])

#     for k in BalloonNetwork:    
#       vl = BalloonNetwork[k]
#       if isinstance(vl, list):
#           vl = np.concatenate(vl, axis=1)
#           BalloonNetwork[k] = vl

#     return BalloonNetwork


def BalloonNetwork_roiwise(timing, unodes, A, p_inst=0.5, use_lag=True, lag_scale=1.0):
    """
    Versión con parte instantánea y parte con lag, similar al SVAR(1) de DCM.

    A      : matriz de conectividad completa (ground truth)
    p_inst : probabilidad de que una conexión de A vaya a C_inst (instantáneo)
    use_lag: si True, aplica también B_lag a la señal neuronal
    """

    # 1) Separar A en parte instantánea y parte con lag
    C_inst, B_lag = split_A_disjoint(A, p_inst=p_inst, seed=42)

    const = 1.0  

    BalloonNetwork = {
        'timing': timing,
        'u': unodes,
        'tneur': [],
        'neur': [],
        'neur_inst': [],   # solo instantáneo
        'neur_eff': [],    # instantáneo + lag
        'tflowin': [],
        'flowin': [],
        'vascsignal': [],
        'tbold': [],
        'bold': [],
        'v': [],
        'q': [],
        # guardamos también las matrices de conectividad
        'Adjacency': A,
        'Adjacency_inst': C_inst,
        'Adjacency_lag': B_lag,
    }

    # 2) Señal neuronal usando U como input directo
    u = unodes.copy()
    tneur, neur = StimulusToNeural(timing, u, const, C_inst)

    BalloonNetwork['tneur'] = tneur
    BalloonNetwork['neur'] = neur
    BalloonNetwork['tu'] = timing

    # 3) Aplicar parte instantánea (C_inst)
    neur_inst = neur + neur @ C_inst.T
    BalloonNetwork['neur_inst'] = neur_inst

    # 4) Aplicar parte con lag (B_lag) sobre neur_inst
    if use_lag:
        neur_eff = add_lag_to_neural(neur_inst, B_lag, scale=lag_scale)
    else:
        neur_eff = neur_inst

    BalloonNetwork['neur_eff'] = neur_eff

    TE = 0.03
    Nsamples, Nrois = neur_eff.shape

    # 5) De neur_eff → flujo → BOLD
    for inode in range(Nrois):
        print(f'node {inode+1}')
        tflowin, flowin, vascsignal = NeuraltoFlow(tneur, neur_eff[:, inode])
        tbold, bold, q, v = BalloonModel(tflowin, flowin, TE)
        BalloonNetwork['tflowin'].append(tflowin[:, np.newaxis])
        BalloonNetwork['flowin'].append(flowin[:, np.newaxis])
        BalloonNetwork['vascsignal'].append(vascsignal[:, np.newaxis])
        BalloonNetwork['tbold'].append(tbold[:, np.newaxis])
        BalloonNetwork['bold'].append(bold[:, np.newaxis])
        BalloonNetwork['v'].append(v[:, np.newaxis])
        BalloonNetwork['q'].append(q[:, np.newaxis])

    # 6) Convertir listas a arrays como antes
    for k in BalloonNetwork:
        vl = BalloonNetwork[k]
        if isinstance(vl, list):
            vl = np.concatenate(vl, axis=1)
            BalloonNetwork[k] = vl

    return BalloonNetwork




def plot_network(BalloonNetwork, ks=['bold'], save = None):
    # # plot it
    Nnodes = BalloonNetwork['Adjacency'].shape[0]
    if 'neural' in ks:
        fig1, ax1 = plt.subplots(Nnodes, 1, figsize=(30,10))
        for inode in range(Nnodes):
            ax1[inode].plot(BalloonNetwork['tneur'],
                     BalloonNetwork['neur'][:, inode])
            ax1[inode].set_ylabel(f'node {inode+1}')
        ax1[0].set_title('neuronal signal')
        ax1[-1].set_xlabel('time')

    if save != None:
        img_filename = "Neuronal signal"
        full_path_img = os.path.join(save, img_filename)
        plt.savefig(full_path_img, dpi=300, bbox_inches='tight')

    if 'flow in' in ks:
        fig2, ax2 = plt.subplots(Nnodes, 2, figsize=(30,10))
        for inode in range(Nnodes):
            ax2[inode, 0].plot(BalloonNetwork['tflowin'][:, inode],
                     BalloonNetwork['vascsignal'][:, inode],
                     label='signal to vasculature')
            ax2[inode, 0].set_ylabel(f'node {inode+1}')

            ax2[inode, 1].plot(BalloonNetwork['tflowin'][:, inode],
                     BalloonNetwork['flowin'][:, inode],
                     label='flow in')
            ax2[inode, 1].set_ylabel(f'n{inode+1}')
        ax2[0, 0].set_title('signal to vasculature')
        ax2[0, 1].set_title('flow in to vasculature')
        ax2[-1, 0].set_xlabel('time')
        ax2[-1, 1].set_xlabel('time')

    if save != None:
        img_filename = "signal to flow in to vasculature"
        full_path_img = os.path.join(save, img_filename)
        plt.savefig(full_path_img, dpi=300, bbox_inches='tight')

    if 'qv' in ks:
        fig3, ax3 = plt.subplots(Nnodes, 2, figsize=(30,10))
        for inode in range(Nnodes):
            ax3[inode, 0].plot(BalloonNetwork['tbold'][:, inode],
                     BalloonNetwork['q'][:, inode],
                     label='q')
            ax3[inode, 0].set_ylabel(f'node {inode+1}')

            ax3[inode, 1].plot(BalloonNetwork['tbold'][:, inode],
                     BalloonNetwork['v'][:, inode],
                     label='v')
            ax3[inode, 1].set_ylabel(f'n{inode+1}')
        ax3[0, 0].set_title('[deoxyhemoglobin]')
        ax3[0, 1].set_title('blood volume')
        ax3[-1, 0].set_xlabel('time')
        ax3[-1, 1].set_xlabel('time')

    if save != None:
        img_filename = "deoxyhemoglobin blood volume"
        full_path_img = os.path.join(save, img_filename)
        plt.savefig(full_path_img, dpi=300, bbox_inches='tight')

    if 'bold' in ks:
        fig4, ax4 = plt.subplots(Nnodes, 1, figsize=(30,10))
        for inode in range(Nnodes):
            ax4[inode].plot(BalloonNetwork['tbold'][:, inode],
                     BalloonNetwork['bold'][:, inode],
                     label='bold')
            ax4[inode].set_ylabel(f'n{inode+1}')
        ax4[0].set_title('bold')
        ax4[-1].set_xlabel('time')

    if save != None:
        img_filename = "BOLD"
        full_path_img = os.path.join(save, img_filename)
        plt.savefig(full_path_img, dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()

def gen_A(N_nds=5, N_conn=3):
    A = np.eye(N_nds)
    node_pairs = []
    for i in range(N_nds):
        for j in range(N_nds):
            if i > j:
                node_pairs.append( (i,j) ) 
    nodes_connected = []
    while len(nodes_connected)<N_conn:
        n = rn.choice( node_pairs )
        if n not in nodes_connected:
            nodes_connected.append( n )
    neg_ratio = 0.3  # 30% de las conexiones serán negativas  ####################
    
    for i, j in nodes_connected:
        #A[i, j] = rn.choice(np.arange(0.5,0.9,0.1)) #intensidad de conexion
        #A[i, j] = rn.choice([-1, 1]) * rn.uniform(0.2, 1.0) # permite conexiones negativas
        sign = -1 if rn.random() < neg_ratio else 1         #########################################
        A[i, j] = sign * rn.uniform(0.2, 1.0)
    return A

def gen_node_stimulus(timeline, nblocks=5, duration_range=(1,3)):
    N_pts = len(timeline)
    time_offset = 10
    possible_onsets = np.arange(time_offset, N_pts-time_offset)
    sep_bwn_ons = 10
    durations = []
    onsets = []    
    for i in range(nblocks):
        current_block_duration = rn.randint(duration_range[0], duration_range[1])
        current_onset = rn.choice( possible_onsets )
        onsets.append( current_onset )
        durations.append( current_block_duration )
        possible_onsets = possible_onsets[np.logical_or( possible_onsets < current_onset - sep_bwn_ons,
                                                possible_onsets > current_onset + sep_bwn_ons)]
    u = np.zeros(N_pts)
    for i,d in zip(onsets, durations):
        u[i:i+d] = 1
    return u, timeline, onsets, durations
    
def gen_network_stimulus(timeline, N_nds, nblocks=5, duration_range=(1,3), seed=None):
    
     N_pts = len(timeline)
     U = np.zeros(shape=(N_pts,N_nds))
    
     for n in range( N_nds ):
         U[:,n],_,_,_ = gen_node_stimulus(timeline, nblocks, duration_range)
     return U
    


# parameters

#Nsamples = 250
#TR = 1
#timeline = TR*np.arange( Nsamples )
#N_nds = 15
#N_conn = 20

#C = gen_A(N_nds, N_conn)
#U = gen_network_stimulus(timeline, N_nds, nblocks=5, duration_range=(1,3))

#plt.figure()
#plt.imshow(C)
#plt.show()


# # run it (roi-wise)
#BalloonNetwork = BalloonNetwork_roiwise(timeline, U, C)
#plot_network(BalloonNetwork, ks=['neural', 'flow in', 'qv', 'bold'])