



import matplotlib.pyplot as plt
import numpy as np
from acados_template import latexify_plot

def plot_results(shooting_nodes, u_max, U, X_true, X_est=None, Y_measured=None, latexify=False, plt_show=True, X_true_label=None,
    time_label='$t$', x_labels=["q"+str(i) for i in range(1, 9)],
    title = None
                  ):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    WITH_ESTIMATION = X_est is not None and Y_measured is not None

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]
    nx_half = nx // 2

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    Ts = t[1] - t[0]
    if WITH_ESTIMATION:
        N_mhe = N_sim - X_est.shape[0]
        t_mhe = np.linspace(N_mhe * Ts, Tf, N_sim-N_mhe)

    # add u0 to u
    u_ = np.vstack((U[0,:], U))
    nu = u_.shape[1]

    for i in range(nu):
        plt.subplot(3, 1, 1)
        line, = plt.step(t, u_[:, i], label='u'+str(i+1))
    if X_true_label is not None:
        line.set_label(X_true_label)
    # else:
    #     line.set_color('r')
    if title is not None:
        plt.title(title)
    plt.ylabel('$u$')
    plt.xlabel(time_label)
    plt.hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.ylim([-1.2*u_max, 1.2*u_max])
    plt.grid()
    plt.legend(loc=1)


    for i in range(nx_half):
        plt.subplot(3, 1, 2)
        line, = plt.plot(t, X_true[:, i], label="q"+str(i+1))
    if X_true_label is not None:
        line.set_label(X_true_label)

    if WITH_ESTIMATION:
        plt.plot(t_mhe, X_est[:, i], '--', label='estimated')
        plt.plot(t, Y_measured[:, i], 'x', label='measured')

    plt.ylabel('$q$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend(loc=1)


    for i in range(nx_half,nx):
        plt.subplot(3, 1, 3)
        line, = plt.plot(t, X_true[:, i], label="qdot"+str(i+1-nx_half))
    if X_true_label is not None:
        line.set_label(X_true_label)

    if WITH_ESTIMATION:
        plt.plot(t_mhe, X_est[:, i], '--', label='estimated')
        plt.plot(t, Y_measured[:, i], 'x', label='measured')

    plt.ylabel('$\dot{q}$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend(loc=1)


    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    if plt_show:
        plt.show(block=False)
    return



def plot_pfc_results(shooting_nodes, u_max, U, X_true, X_est=None, Y_measured=None, latexify=False, plt_show=True, X_true_label=None,
    time_label='$t$', x_labels=["q"+str(i) for i in range(1, 9)],
    title = None, linestyle='solid'
                  ):
    """
    Params:
        shooting_nodes: time values of the discretization
        u_max: maximum absolute value of u
        U: arrray with shape (N_sim-1, nu) or (N_sim, nu)
        X_true: arrray with shape (N_sim, nx)
        X_est: arrray with shape (N_sim-N_mhe, nx)
        Y_measured: array with shape (N_sim, ny)
        latexify: latex style plots
    """

    if latexify:
        latexify_plot()

    WITH_ESTIMATION = X_est is not None and Y_measured is not None

    N_sim = X_true.shape[0]
    nx = X_true.shape[1]
    nq = (nx-1) // 2

    Tf = shooting_nodes[N_sim-1]
    t = shooting_nodes

    Ts = t[1] - t[0]
    if WITH_ESTIMATION:
        N_mhe = N_sim - X_est.shape[0]
        t_mhe = np.linspace(N_mhe * Ts, Tf, N_sim-N_mhe)

    # add u0 to u
    u_ = np.vstack((U[0,:], U))
    nu = u_.shape[1]

    for i in range(nu-1):
        plt.subplot(4, 1, 1)
        line, = plt.step(t, u_[:, i], label='u'+str(i+1), linestyle=linestyle)
    if X_true_label is not None:
        line.set_label(X_true_label)
    # else:
    #     line.set_color('r')
    if title is not None:
        plt.title(title)
    plt.ylabel('$u$')
    plt.xlabel(time_label)
    plt.hlines(u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.hlines(-u_max, t[0], t[-1], linestyles='dashed', alpha=0.7)
    plt.ylim([-1.2*u_max, 1.2*u_max])
    plt.grid()
    plt.legend(loc=1)


    for i in range(nq):
        plt.subplot(4, 1, 2)
        line, = plt.plot(t, X_true[:, i], label="q"+str(i+1), linestyle=linestyle)
    if X_true_label is not None:
        line.set_label(X_true_label)

    if WITH_ESTIMATION:
        plt.plot(t_mhe, X_est[:, i], '--', label='estimated')
        plt.plot(t, Y_measured[:, i], 'x', label='measured')

    plt.ylabel('$q$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend(loc=1)


    for i in range(nq,2*nq):
        plt.subplot(4, 1, 3)
        line, = plt.plot(t, X_true[:, i], label="qdot"+str(i+1-nq), linestyle=linestyle)
    if X_true_label is not None:
        line.set_label(X_true_label)

    if WITH_ESTIMATION:
        plt.plot(t_mhe, X_est[:, i], '--', label='estimated')
        plt.plot(t, Y_measured[:, i], 'x', label='measured')

    plt.ylabel('$\dot{q}$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend(loc=1)

    ## path parameter 

    plt.subplot(4, 1, 4)
    line, = plt.plot(t, X_true[:, nx-2], label="theta", linestyle=linestyle)
    line, = plt.plot(t, X_true[:, nx-1], label="theta_dot", linestyle=linestyle)
    line, = plt.plot(t, u_[:, nu-1], label="v", linestyle=linestyle)
    plt.ylabel('$theta$, $theta_dot$, $v$')
    plt.xlabel('$t$')
    plt.grid()
    plt.legend(loc=1)

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, hspace=0.4)

    if plt_show:
        plt.show()
    else:
        plt.show(block=False)
    return


def close_all():
    """
    Close all matplotlib figures.
    """
    plt.close('all')
    return