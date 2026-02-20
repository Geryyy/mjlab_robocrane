from acados_template import AcadosModel
from casadi import SX, vertcat, sin, cos, Function
import casadi as ca
from pinocchio import casadi as cpin

def export_robocrane_ode_model(cpin_model, cpin_data, tool_frame_id, tool_body_id=-1) -> AcadosModel:

    model_name = 'robocrane'
    
    nq = cpin_model.nq  # number of generalized coordinates
    nua = 2             # under actuated joints
    nx = 2 * nq         # State dimension (q and qp)
    nu = nq - nua    # Control input dimension

    # Create symbolic variables for states and controls
    q = SX.sym('q', nq)       # Actuated joint positions
    qp = SX.sym('qp', nq)     # Actuated joint velocities
    u = SX.sym('u', nu)       # Control inputs

    tool_mass = SX.sym('tool_mass', 1)
    tool_com = SX.sym('tool_com', 3) # Center of mass [x, y, z]

    # Full state
    x = vertcat(q, qp)

    # Dynamics: create qpp_u similar to your implementation
    M = cpin.crba(cpin_model, cpin_data, q)
    G = cpin.computeGeneralizedGravity(cpin_model, cpin_data, q)
    # nle = cpin.nonLinearEffects(cpin_model, cpin_data, q, qp)
    Mu = M[7:, 7:]
    Mua = M[7:, :7]
    # nleu = nle[7:]
    damping_u = ca.diag(cpin_model.damping[7:])
    G_u = G[7:]
    
    # qpp_u = casadi.inv(Mu) @ (- Mua @ qpp_a - nleu - damping_u @ qp[7:])
    # qpp_u = ca.inv(Mu) @ (- Mua @ u - G_u - damping_u @ qp[7:])

    # Modify inertia and gravity based on tool mass and CoG
    if tool_body_id > 0 and tool_body_id < cpin_model.nbodies:
        # Get the original inertia of the tool frame
        original_inertia = cpin_model.inertias[tool_body_id]

        # Create a new inertia object with the parameterized mass and CoG
        modified_inertia = cpin.Inertia(tool_mass, tool_com, original_inertia.inertia)

        # Update the inertia of the tool frame in a copy of the model
        modified_model = cpin_model.copy()
        modified_model.inertias[tool_body_id] = modified_inertia
        modified_data = modified_model.createData()

        # Recalculate M and G with the modified model
        M_modified = cpin.crba(modified_model, modified_data, q)
        G_modified = cpin.computeGeneralizedGravity(modified_model, modified_data, q)
        nle_modified = cpin.nonLinearEffects(modified_model, modified_data, q, qp)
        Mu_modified = M_modified[7:, 7:]
        Mua_modified = M_modified[7:, :7]
        G_u_modified = G_modified[7:]
        nle_u_modified = nle_modified[7:]

        qpp_u = ca.inv(Mu_modified) @ (- Mua_modified @ u - G_u_modified - damping_u @ qp[7:])
        # qpp_u = ca.inv(Mu_modified) @ (- Mua_modified @ u - nle_u_modified - damping_u @ qp[7:])
    else:
        print(f"Warning: Tool frame with ID '{tool_frame_id}' not found in the Pinocchio model. Using original dynamics.")
        qpp_u = ca.inv(Mu) @ (- Mua @ u - G_u - damping_u @ qp[7:])
    
    # State dynamics f(x,u)
    f_expl = vertcat(qp, vertcat(u, qpp_u))

    # qpp = SX.sym('qpp', nq)
    # xdot = vertcat(qp, qpp)
    xdot = SX.sym('xdot', nx, 1)
    f_impl = xdot - f_expl

    # parameters
    p = vertcat(tool_mass, tool_com.reshape((-1, 1)))

    pval_mass = 1*np.ones(1)
    pval_com = np.zeros(3)
    p_val = np.concatenate((pval_mass, pval_com))   

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    # model.z = z
    model.p = p
    model.name = model_name

    return model, p_val


import numpy as np
def create_param_vector(nq, n_ctr_pts):
    CtrPts = SX.sym('CtrPts', n_ctr_pts*nq)
    q0 = np.zeros(nq)
    q1 = np.ones(nq)
    c_val = (np.linspace(q0, q1, n_ctr_pts))

    param = CtrPts
    param_val = np.reshape(c_val, (n_ctr_pts * nq), 'F')
    return param, param_val


def create_taskspace_param_vector(nts, nq, n_ctr_pts):
    CtrPtsTs = SX.sym('CtrPtsTs', n_ctr_pts*nts)
    p0 = np.zeros(nts)
    p1 = np.zeros(nts)
    c_ts = (np.linspace(p0, p1, n_ctr_pts))
    param_val_ts = np.reshape(c_ts, (n_ctr_pts * nts), 'F')
    
    CtrPtsJs = SX.sym('CtrPtsJs', n_ctr_pts*nq)
    q0 = np.zeros(nq)
    q1 = np.ones(nq)
    c_js = (np.linspace(q0, q1, n_ctr_pts))
    param_val_js = np.reshape(c_js, (n_ctr_pts * nq), 'F')
    
    param = ca.vertcat(CtrPtsTs, CtrPtsJs)
    param_val = np.concatenate([param_val_ts, param_val_js])
    return param, param_val




def export_robocrane_pfc_model(cpin_model, cpin_data, tool_frame_id, tool_body_id=-1, n_ctrl_pts=3, taskspace_flag=False) -> AcadosModel:
    model_name = 'robocrane_pfc'
    nq = cpin_model.nq  # number of generalized coordinates
    nua = 2             # under actuated joints
    nx = 2 * nq        # State dimension (q and qp)
    nx_bar = nx + 1 + 1     # State dimension (q and qp and theta and theta_dot)
    nu = nq - nua     # Control input dimension (u)
    nu_bar = nu + 1     # Control input dimension (u and theta_dotdot)
    nts = 3 + 1         # Task space dimension (x, y, z, yaw)
    
    # Create symbolic variables for states and controls
    q = SX.sym('q', nq)       # Actuated joint positions
    qp = SX.sym('qp', nq)     # Actuated joint velocities
    theta = SX.sym('theta', 1) # path parameter
    theta_dot = SX.sym('theta_dot', 1) # path parameter
    x = vertcat(q, qp)

    # inputs
    u = SX.sym('u', nu)       # Control inputs
    v = SX.sym('v', 1)        # Control input for path parameter v = theta_dotdot
    
    # Full state
    x_bar = vertcat(x,theta, theta_dot)
    u_bar = vertcat(u,v)
    
    # Parameter of model
    tool_mass = SX.sym('tool_mass', 1)
    tool_com = SX.sym('tool_com', 3) # Center of mass [x, y, z]

    # parameter of cost function
    weight_pos = SX.sym('weight_pos', 3)
    weight_rot = SX.sym('weight_rot', 1)
    weight_q = SX.sym('weight_q', 9)
    weight_qp = SX.sym('weight_qp', 9)
    weight_theta = SX.sym('weight_theta', 1)
    weight_theta_dot = SX.sym('weight_theta_dot', 1)
    weight_u = SX.sym('weight_u', 7)
    weight_v = SX.sym('weight_v', 1)
    theta_ref = SX.sym('theta_ref', 1)  # Reference for theta

    # Dynamics: create qpp_u similar to your implementation
    M = cpin.crba(cpin_model, cpin_data, q)
    G = cpin.computeGeneralizedGravity(cpin_model, cpin_data, q)
    nle = cpin.nonLinearEffects(cpin_model, cpin_data, q, qp)
    Mu = M[7:, 7:]
    Mua = M[7:, :7]
    # nleu = nle[7:]
    damping_u = ca.diag(cpin_model.damping[7:])
    G_u = G[7:]
    nle_u = nle[7:]

    # Modify inertia and gravity based on tool mass and CoG
    if tool_body_id > 0 and tool_body_id < cpin_model.nbodies:
        # Get the original inertia of the tool frame
        original_inertia = cpin_model.inertias[tool_body_id]

        # Create a new inertia object with the parameterized mass and CoG
        modified_inertia = cpin.Inertia(tool_mass, tool_com, original_inertia.inertia)

        # Update the inertia of the tool frame in a copy of the model
        modified_model = cpin_model.copy()
        modified_model.inertias[tool_body_id] = modified_inertia
        modified_data = modified_model.createData()

        # Recalculate M and G with the modified model
        M_modified = cpin.crba(modified_model, modified_data, q)
        G_modified = cpin.computeGeneralizedGravity(modified_model, modified_data, q)
        nle_modified = cpin.nonLinearEffects(modified_model, modified_data, q, qp)
        Mu_modified = M_modified[7:, 7:]
        Mua_modified = M_modified[7:, :7]
        G_u_modified = G_modified[7:]
        nle_u_modified = nle_modified[7:]

        qpp_u = ca.inv(Mu_modified) @ (- Mua_modified @ u - G_u_modified - damping_u @ qp[7:])
        # qpp_u = ca.inv(Mu_modified) @ (- Mua_modified @ u - nle_u_modified - damping_u @ qp[7:])
    else:
        print(f"Warning: Tool frame with ID '{tool_frame_id}' not found in the Pinocchio model. Using original dynamics.")
        qpp_u = ca.inv(Mu) @ (- Mua @ u - G_u - damping_u @ qp[7:])
    
    
    # State dynamics f(x,u)
    f_expl = vertcat(qp, vertcat(u, qpp_u), theta_dot, v)

    xdot = SX.sym('xdot', nx_bar, 1)
    f_impl = xdot - f_expl
    
    # parameters
    n_param_fixed = 1 + 3 # mass + com_x + com_y + com_z
    if taskspace_flag:
        p_ctrl_pts, p_ctrl_pts_val = create_taskspace_param_vector(nts, nq, n_ctrl_pts)
    else:
        p_ctrl_pts, p_ctrl_pts_val = create_param_vector(nq, n_ctrl_pts)

    # Add symbolic parameters for tool mass and CoG
    p = vertcat(weight_pos, weight_rot, 
                weight_q, weight_qp, weight_theta, weight_theta_dot,
                weight_u, weight_v, 
                tool_mass, tool_com.reshape((-1, 1)), 
                theta_ref,
                p_ctrl_pts)

    pval_weight_pos = 10*np.ones(3)
    pval_weight_rot = 0.1*np.ones(1)

    pval_weight_q = np.concatenate([5*np.ones(7), 0.1*np.ones(2)])
    pval_weight_qp = np.concatenate([0.1*np.ones(7), 5.0*np.ones(2)])
    pval_weight_theta = 10*np.ones(1)
    pval_weight_theta_dot = 10*np.ones(1)

    pval_weight_u = 2*np.ones(7)
    pval_weight_v = 10*np.ones(1)

    pval_mass = 1*np.ones(1)
    pval_com = np.zeros(3)

    pval_theta_ref = np.ones(1)

    p_val = np.concatenate((pval_weight_pos, pval_weight_rot, 
                            pval_weight_q, pval_weight_qp, pval_weight_theta, pval_weight_theta_dot,
                            pval_weight_u, pval_weight_v,  
                            pval_mass, pval_com, pval_theta_ref,
                            p_ctrl_pts_val)) 

    model = AcadosModel()

    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = x_bar
    model.xdot = xdot
    model.u = u_bar
    # model.z = z
    model.p = p
    model.name = model_name

    return model, p_val


def export_linearized_robocrane(cpin_model, cpin_data, tool_frame_id, xbar, ubar):
    model = export_robocrane_ode_model(cpin_model, cpin_data, tool_frame_id)

    val = ca.substitute(ca.substitute(model.f_expl_expr, model.x, xbar), model.u, ubar)
    jac_x = ca.substitute(ca.substitute(ca.jacobian(model.f_expl_expr, model.x), model.x, xbar), model.u, ubar)
    jac_u = ca.substitute(ca.substitute(ca.jacobian(model.f_expl_expr, model.u), model.x, xbar), model.u, ubar)

    model.f_expl_expr = val + jac_x @ (model.x-xbar) + jac_u @ (model.u-ubar)
    model.f_impl_expr = model.f_expl_expr - model.xdot
    model.name += '_linearized'
    return model


def export_robocrane_ode_model_with_discrete_rk4(cpin_model, cpin_data, tool_frame_id, dT):

    model = export_robocrane_ode_model(cpin_model, cpin_data, tool_frame_id)

    x = model.x
    u = model.u

    ode = Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1,u)
    k3 = ode(x+dT/2*k2,u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_linearized_pendulum_ode_model_with_discrete_rk4(cpin_model, cpin_data, tool_frame_id, dT, xbar, ubar):

    model = export_linearized_pendulum(cpin_model, cpin_data, tool_frame_id, xbar, ubar)

    x = model.x
    u = model.u

    ode = Function('ode', [x, u], [model.f_expl_expr])
    # set up RK4
    k1 = ode(x,       u)
    k2 = ode(x+dT/2*k1,u)
    k3 = ode(x+dT/2*k2,u)
    k4 = ode(x+dT*k3,  u)
    xf = x + dT/6 * (k1 + 2*k2 + 2*k3 + k4)

    model.disc_dyn_expr = xf
    # print("built RK4 for pendulum model with dT = ", dT)
    # print(xf)
    return model

def export_augmented_pendulum_model(cpin_model, cpin_data, tool_frame_id):
    # pendulum model augmented with algebraic variable just for testing
    model = export_robocrane_ode_model(cpin_model, cpin_data, tool_frame_id)
    model_name = 'augmented_pendulum'

    z = SX.sym('z', 2, 1)

    f_impl = vertcat( model.xdot - model.f_expl_expr, \
        z - vertcat(model.x[0], model.u**2)
    )

    model.f_impl_expr = f_impl
    model.z = z
    model.name = model_name

    return model

