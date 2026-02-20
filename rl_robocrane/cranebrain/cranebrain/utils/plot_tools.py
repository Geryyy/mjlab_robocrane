import matplotlib.pyplot as plt


def plot_trajectory(tt, q_vec, qp_vec, u_vec, dof, dof_a, t0, q0, qp0, qd, N_prev=10, fig=None, axs=None, previous_plots=None):
    # Initialize previous_plots if it's the first call
    if previous_plots is None:
        previous_plots = {
            'q': [],
            'qp': [],
            'u': []
        }

    init = not all(previous_plots.values())

    # Use a more distinctive colormap
    colors = plt.get_cmap('tab10').colors

    # Define fainter colors
    faint_color = '0.7'  # Light gray

    # Initialize figure and axes if it's the first call
    if fig is None or axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(8, 12))

    # Plot q
    if init:
        axs[0].set_title('Joint State (q)')
    for i in range(dof):
        # Plot the initial state q0 with a distinct marker
        axs[0].plot(t0, q0[i], 'o', label='q_init' if i == 0 else "", color=colors[i % len(colors)])

        line, = axs[0].plot(tt, q_vec[:, i], label='q' + str(i), color=colors[i % len(colors)])
        # if i == dof-2 or i == dof-1:
        #     # line.set_linestyle('--')
        #     line.set_linewidth(5)

        previous_plots['q'].append(line)
       

    # Plot qw and qd with distinct colors and linestyles
    for i in range(dof):
        # axs[0].axhline(y=qw[i], color='orange', linestyle='--', label='qw' + str(i) if i == 0 else "")
        axs[0].axhline(y=qd[i], color='green', linestyle='-.', label='qd' + str(i) if i == 0 else "")

    axs[0].set_xlabel('t in s')
    axs[0].set_ylabel('q in rad')
    axs[0].grid(True)

    if init:
        axs[0].legend()

    # Plot qp
    if init:
        axs[1].set_title('Joint Velocity (qp)')
    for i in range(dof):
        # Plot the initial velocity qp0 with a distinct marker
        axs[1].plot(t0, qp0[i], 'o', label='qp_init' if i == 0 else "", color=colors[i % len(colors)])

        line, = axs[1].plot(tt, qp_vec[:, i], label='qp' + str(i), color=colors[i % len(colors)])
        if i == dof-2 or i == dof-1:
            # line.set_linestyle('dashdot')
            line.set_linewidth(5)

        previous_plots['qp'].append(line)
        
    
    axs[1].set_xlabel('t in s')
    axs[1].set_ylabel('qp in rad/s')
    axs[1].grid(True)
    
    if init:
        axs[1].legend()


    # Plot u
    if init:
        axs[2].set_title('Control Input (u)')
    for i in range(dof_a):
        line, = axs[2].plot(tt, u_vec[:, i], label='u' + str(i), color=colors[i % len(colors)])
        previous_plots['u'].append(line)
    axs[2].set_xlabel('t in s')
    axs[2].set_ylabel('u in rad/s^2')
    axs[2].grid(True)
    
    if init:
        axs[2].legend()

    # Remove oldest entry if number of entries is greater than N_prev

    # Remove oldest entry if number of entries is greater than N_prev
    for key, ax in zip(previous_plots.keys(), axs):
        if key != 'u' and len(previous_plots[key]) > N_prev * dof:
            for _ in range(dof):
                if len(previous_plots[key]) > 0:
                    oldest_line = previous_plots[key].pop(0)
                    ax.lines.remove(oldest_line)
                    # print("removing oldest line for key:", key)
        elif key == 'u' and len(previous_plots[key]) > N_prev * dof_a:
            for _ in range(dof_a):
                if len(previous_plots[key]) > 0:
                    oldest_line = previous_plots[key].pop(0)
                    ax.lines.remove(oldest_line)
                    # print("removing oldest line for key:", key)

    # Update previous plots to dashed lines with fainter color
    for key in previous_plots:
        for line in previous_plots[key][:-dof if key != 'u' else -dof_a]:
            line.set_linestyle('--')
            # line.set_color(faint_color)
            line.set_alpha(0.5)

    # Update x-axis limits to include the maximum value of tt
    max_tt = max(tt)
    for ax in axs:
        ax.set_xlim(right=max_tt)

    # Redraw the figure
    fig.canvas.draw()
    plt.show(block=False)

    return fig, axs, previous_plots






def plot_acceleration_error(tt, model_acc_vec, sim_acc_vec, e_acc_vec, N_prev=10, fig=None, axs=None, previous_plots=None):
    # Initialize previous_plots if it's the first call
    if previous_plots is None:
        previous_plots = {
            'model_acc': [],  # 'model_acc' is the acceleration computed by the model
            'sim_acc': [],  # 'sim_acc' is the acceleration computed by the simulator
            'e_acc': []
        }

    init = not any(previous_plots.values())

    # Use a more distinctive colormap
    colors = plt.get_cmap('tab10').colors

    # Define fainter colors
    faint_color = '0.7'  # Light gray

    # Initialize figure and axes if it's the first call
    if fig is None or axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(8, 18))

    # Plot model_acc
    if init:
        axs[0].set_title('Model Acceleration (model_acc)')
    for i in range(model_acc_vec.shape[1]):
        line, = axs[0].plot(tt, model_acc_vec[:, i], label='model_acc' + str(i), color=colors[i % len(colors)])
        if i < 7:
            line.set_linestyle('--')
            line.set_alpha(0.5)
        previous_plots['model_acc'].append(line)
    axs[0].set_xlabel('t in s')
    axs[0].set_ylabel('Acceleration')
    axs[0].grid(True)
    if init:
        # legend on the right side
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot sim_acc
    if init:
        axs[1].set_title('Simulator Acceleration (sim_acc)')
    for i in range(sim_acc_vec.shape[1]):
        line, = axs[1].plot(tt, sim_acc_vec[:, i], label='sim_acc' + str(i), color=colors[i % len(colors)])
        if i < 7:
            line.set_linestyle('--')
            line.set_alpha(0.5)
        previous_plots['sim_acc'].append(line)
    axs[1].set_xlabel('t in s')
    axs[1].set_ylabel('Acceleration')
    axs[1].grid(True)
    if init:
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot e_acc
    if init:
        axs[2].set_title('Acceleration Error (e_acc)')
    for i in range(e_acc_vec.shape[1]):
        line, = axs[2].plot(tt, e_acc_vec[:, i], label='e_acc' + str(i), color=colors[i % len(colors)])
        if i < 7:
            line.set_linestyle('--')
            line.set_alpha(0.5)
        previous_plots['e_acc'].append(line)
    axs[2].set_xlabel('t in s')
    axs[2].set_ylabel('Error')
    axs[2].grid(True)
    if init:
        axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Remove oldest entry if number of entries is greater than N_prev
    for key, ax in zip(previous_plots.keys(), axs):
        if len(previous_plots[key]) > N_prev * len(e_acc_vec[0]):
            for _ in range(len(e_acc_vec[0])):
                if len(previous_plots[key]) > 0:
                    oldest_line = previous_plots[key].pop(0)
                    ax.lines.remove(oldest_line)

    # Update previous plots to dashed lines with fainter color
    for key in previous_plots:
        for line in previous_plots[key][:-len(e_acc_vec[0])]:
            line.set_linestyle('--')
            line.set_alpha(0.5)

    # Update x-axis limits to include the maximum value of tt
    max_tt = max(tt)
    min_tt = min(tt)
    # enuermate axs
    for i, ax in enumerate(axs):
        ax.set_xlim(left=min_tt, right=max_tt)
        if i == 0:
            ax.set_ylim(bottom=min(model_acc_vec.flatten()), top=max(model_acc_vec.flatten()))
        if i == 1:
            ax.set_ylim(bottom=min(sim_acc_vec.flatten()), top=max(sim_acc_vec.flatten()))
        if i == 2:
            ax.set_ylim(bottom=min(e_acc_vec.flatten()), top=max(e_acc_vec.flatten()))



    # autoscale y-axis for axs[2]
    # axs[2].relim()
    # axs[2].autoscale_view()


    # Redraw the figure
    fig.canvas.draw()
    plt.show(block=False)

    return fig, axs, previous_plots





def plot_setpoint(tt, q_set, qp_set, qpp_set, t0, q0, qp0, fig=None, axs=None, previous_plots=None):

    init = fig==None

    # Use a more distinctive colormap
    colors = plt.get_cmap('tab10').colors

    # Define fainter colors
    faint_color = '0.7'  # Light gray

    # Initialize figure and axes if it's the first call
    if fig is None or axs is None:
        fig, axs = plt.subplots(3, 1, figsize=(8, 18))

    # Plot model_acc
    if init:
        axs[0].set_title('q_set')

    for i in range(q_set.shape[1]):
        axs[0].plot(t0, q0[i], 'o', label='q_init' if i == 0 and init else "", color=colors[i % len(colors)])
        line, = axs[0].plot(tt, q_set[:, i], label='q_set' + str(i) if init else "", color=colors[i % len(colors)])


    axs[0].set_xlabel('t in s')
    axs[0].set_ylabel('q_set')
    axs[0].grid(True)
    if init:
        # legend on the right side
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot qp_set
    if init:
        axs[1].set_title('qp_set')

    for i in range(qp_set.shape[1]):
        axs[1].plot(t0, qp0[i], 'o', label='qp_init' if i == 0 and init else "", color=colors[i % len(colors)])
        line, = axs[1].plot(tt, qp_set[:, i], label='qp_set' + str(i) if init else "", color=colors[i % len(colors)])
    

    axs[1].set_xlabel('t in s')
    axs[1].set_ylabel('qp_set')
    axs[1].grid(True)
    if init:
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Plot e_acc
    if init:
        axs[2].set_title('qpp_set')
        for i in range(qpp_set.shape[1]):
            line, = axs[2].plot(tt, qpp_set[:, i], label='qpp_set' + str(i), color=colors[i % len(colors)])
    else:
        for i in range(qpp_set.shape[1]):
            line, = axs[2].plot(tt, qpp_set[:, i], color=colors[i % len(colors)])

    axs[2].set_xlabel('t in s')
    axs[2].set_ylabel('qpp_set')
    axs[2].grid(True)
    if init:
        axs[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Update x-axis limits to include the maximum value of tt
    max_tt = max(tt)
    min_tt = min(tt)

    for i, ax in enumerate(axs):
        ax.set_xlim(left=0, right=max_tt)
        # if i == 0:
        #     ax.set_ylim(bottom=min(q_set.flatten()), top=max(q_set.flatten()))
        # if i == 1:
        #     ax.set_ylim(bottom=min(qp_set.flatten()), top=max(qp_set.flatten()))
        # if i == 2:
        #     ax.set_ylim(bottom=min(qpp_set.flatten()), top=max(qpp_set.flatten()))


    # Redraw the figure
    fig.canvas.draw()
    plt.show(block=False)

    return fig, axs, previous_plots

