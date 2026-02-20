import numpy as np
import casadi as ca
# from sspp import _sspp as sp

## BSpline basis function scipy help, can be implemented in casadi
# evaluated at theta
# k..spline order
# i..index
# t..knot points
##
def B(theta, k, i, t):
  if k == 0:
    #return 1.0 if t[i] <= theta < t[i+1] else 0.0
    if t[i] <= theta < t[i+1]:
      return 1.0
    else:
      return 0.0
    #if i == len(t) - k - 2 and theta == t[i+1]:
    #  return 1.0
  if t[i+k] == t[i]:
    c1 = 0.0
  else:
    c1 = (theta - t[i])/(t[i+k] - t[i]) * B(theta, k-1, i, t)
  if t[i+k+1] == t[i+1]:
    c2 = 0.0
  else:
    c2 = (t[i+k+1] - theta)/(t[i+k+1] - t[i+1]) * B(theta, k-1, i+1, t)

  return c1 + c2

def dB(theta, k, i, t):
  if k == 0:
    return 0.0
  if t[i+k] == t[i]:
    c1 = 0.0
  else:
    c1 = k / (t[i+k] - t[i]) * B(theta, k-1, i, t)
  if t[i+k+1] == t[i+1]:
    c2 = 0.0
  else:
    c2 = -k / (t[i+k+1] - t[i+1]) * B(theta, k-1, i+1, t)
  return c1 + c2

def bspline(theta, t, c, k):
  n = len(t) - k - 1
  #assert (n >= k+1) and (len(c) >= n)
  if theta < 0:
    return c[0] * B(0, k, 0, t)
  if theta >= 1:
    return c[n - 1]# * B(1, k, n - 1, t)
  return sum(c[i] * B(theta, k, i, t) for i in range(n))

def bspline_derivative(theta, t, c, k):
  n = len(t) - k - 1
  return sum(c[i] * dB(theta, k, i, t) for i in range(n))


def knot_vector(n_control_points, k):
  n_knots = n_control_points + k + 1  # Total number of knots for B-spline
  t = np.linspace(0, 1, n_knots - 2 * k)  # Internal knots only
  t = np.concatenate(([0] * k, t, [1] * k))  # Add k repeated knots at the ends
  return t


def compute_control_points(via_points, k):
  """
  Computes the control points given a set of via points and the spline order k.
  
  Parameters:
  via_points (ndarray): An array of shape (n, d) where n is the number of via points 
                        and d is the dimensionality of the points (e.g., 2D, 3D).
  k (int): The order of the B-spline.
  
  Returns:
  ndarray: Control points corresponding to the via points.
  ndarray: Knot vector for the B-spline.
  """
  n_via_points = len(via_points)
  n_control_points = n_via_points
  
  # Generate the knot vector using the existing knot_vector function
  t = knot_vector(n_control_points, k)
  
  # Set up the system of linear equations to solve for control points
  A = np.zeros((n_via_points, n_control_points))
  for i in range(n_via_points):
    for j in range(n_control_points):
      A[i, j] = B(i / (n_via_points - 1), k, j, t)
      # print("---")
      # print("i: ", i, "j: ", j)
      # print("theta: ", i / (n_via_points - 1))
      # print("index j: ", j)
      # print("B: ", B(i / (n_via_points - 1), k, j, t))
  A[0, 0] = 1.0
  A[n_via_points - 1, n_control_points - 1] = 1.0
  
  # Solve the system of equations for control points for each dimension of the via points
  control_points = np.linalg.lstsq(A, via_points, rcond=None)[0]

  np.set_printoptions(precision=2, suppress=True)
  # print("A: \n", A)
  # print("control_points: \n", control_points)
  # remove the last control point and repeat the second to last via point
  # control_points = np.vstack((control_points[:-1], control_points[-2]))
  
  return control_points, t



def sample_bspline(c, k=2, n=100):
    """
    Evaluate a B-spline defined by control points c using uniform knot vector.

    Args:
        c: (N, D) array of control points
        k: spline degree (default: 2)
        n: number of samples
    Returns:
        (n, D) array of sampled points along the spline
    """
    c = np.asarray(c)
    t = knot_vector(len(c), k)
    theta_vals = np.linspace(0.0, 1.0, n)
    return np.array([bspline(theta, t, c, k) for theta in theta_vals])



##
# Computes the SLERP interpolation starting from R0 at theta
# S...skew symmetric matrix, phi rotation angle
##
def evalRotationInterpolation(R0, theta, S, phi):
  return np.dot(R0, np.eye(3) + np.sin(theta * phi) * S + (1 - np.cos(theta * phi)) * np.dot(S, S))

## SLERP derivative
def evalRotationInterpolationDiff(R0, theta, S, phi):
  return np.dot(R0, np.cos(theta * phi) * S + np.sin(theta * phi) * np.dot(S, S))

##
# Computes the SLERP interpolation for a list of rotation matrices R at theta
##
def evalRotationInterpolationFull(R, phi, S, theta, theta_vec):
  for i in range(len(R) - 1):
    if theta < theta_vec[i + 1] and theta >= theta_vec[i]:
      return evalRotationInterpolation(R[i], (theta - theta_vec[i]) / (theta_vec[i + 1] - theta_vec[i]), S[i], phi[i]) # theta scaling is important

def evalRotationInterpolationDiffFull(R, phi, S, theta, theta_vec):
  for i in range(len(R) - 1):
    if theta < theta_vec[i + 1] and theta >= theta_vec[i]:
      return evalRotationInterpolationDiff(R[i], (theta - theta_vec[i]) / (theta_vec[i + 1] - theta_vec[i]), S[i], phi[i]) # theta scaling is important

################################################################################################
# casadi versions
################################################################################################

def casadiBspline(theta, t, c, k):
    n = t.shape[0] - k - 1
    
    # Clamp theta to [0, 1] for evaluation
    theta_clamped = ca.fmax(0, ca.fmin(1, theta))
    
    result = 0
    for i in range(n):
        result += c[i] * casadiB(theta_clamped, k, i, t)
    
    return result

def casadiBspline_derivative(theta, t, c, k):
  n = t.shape[0] - k - 1

  # Clamp theta to [0, 1] for evaluation
  theta_clamped = ca.fmax(0, ca.fmin(1, theta))

  result = 0
  for i in range(n):
    result += c[i] * casadi_dB(theta_clamped, k, i, t)

  return result


def casadiBsplineVec(theta, t, c, k):
    n = t.shape[0] - k - 1
    
    # Clamp theta to [0, 1] for evaluation
    theta_clamped = ca.fmax(0, ca.fmin(1, theta))
    
    result = 0
    for i in range(n):
        result += c[i, :] * casadiB(theta_clamped, k, i, t)
    
    return result

def casadiBsplineVec_derivative(theta, t, c, k):
    n = t.shape[0] - k - 1
    
    # Clamp theta to [0, 1] for evaluation
    theta_clamped = ca.fmax(0, ca.fmin(1, theta))
    
    result = 0
    for i in range(n):
        result += c[i, :] * casadi_dB(theta_clamped, k, i, t)
    
    return result


def casadiB(theta, k, i, t):
    if k == 0:
        return ca.if_else(ca.logic_or(ca.logic_and(t[i] <= theta, theta < t[i+1]), theta == t[i + 1]), 1.0, 0.0)
    
    c1 = ca.if_else(t[i+k] == t[i], 0.0, (theta - t[i]) / (t[i+k] - t[i]) * casadiB(theta, k-1, i, t))
    c2 = ca.if_else(t[i+k+1] == t[i+1], 0.0, (t[i+k+1] - theta) / (t[i+k+1] - t[i+1]) * casadiB(theta, k-1, i+1, t))
    return c1 + c2


def casadi_dB(theta, k, i, t):
  if k == 0:
    return 0.0
  if t[i+k] == t[i]:
    c1 = 0.0
  else:
    c1 = k / (t[i+k] - t[i]) * casadiB(theta, k-1, i, t)
  if t[i+k+1] == t[i+1]:
    c2 = 0.0
  else:
    c2 = -k / (t[i+k+1] - t[i+1]) * casadiB(theta, k-1, i+1, t)
  return c1 + c2




def test_bspline():
  import matplotlib.pyplot as plt

  # Step 1: Initialize control points
  q0 = np.zeros(7)
  q1 = np.ones(7)
  c = np.linspace(q0, q1, 7)  # 7 control points from np.zeros(7) to np.ones(7)
  # Step 2: Generate the knot vector
  n_control_points = len(c)  # Number of control points
  k = 3  # Spline order
  
  

  theta = np.linspace(0, 1, 10) # evaluation points
  y = np.array([bspline(x, t, c, k) for x in theta])
  plt.plot(theta, y)
  plt.grid()
  plt.show()



import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


def test_bspline_simple():
    """Simple test for B-spline continuity and derivatives"""
    import matplotlib.pyplot as plt
    import numpy as np
    from sspp import BSplines as bs
    
    # Setup
    c = np.array([0, 1, 2, 1, 0])  # 5 control points
    k = 3
    t = bs.knot_vector(len(c), k)
    
    # Test range including outside [0,1]
    theta = np.linspace(-0.2, 1.2, 100)
    
    # Evaluate spline and derivative
    y = [float(casadiBspline(x, t, c, k)) for x in theta]
    dy = [float(casadiBspline_derivative(x, t, c, k)) for x in theta]
    
    # Check continuity at boundaries
    eps = 1e-8
    print(f"Continuity at θ=0: {abs(float(casadiBspline(-eps, t, c, k)) - float(casadiBspline(0, t, c, k))):.2e}")
    print(f"Continuity at θ=1: {abs(float(casadiBspline(1+eps, t, c, k)) - float(casadiBspline(1, t, c, k))):.2e}")
    
    # Check derivative accuracy at θ=0.5
    theta_test = 0.5
    dy_analytical = float(casadiBspline_derivative(theta_test, t, c, k))
    dy_numerical = (float(casadiBspline(theta_test + eps, t, c, k)) - 
                   float(casadiBspline(theta_test - eps, t, c, k))) / (2*eps)
    print(f"Derivative error at θ=0.5: {abs(dy_analytical - dy_numerical):.2e}")
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.plot(theta, y, 'b-', linewidth=2)
    ax1.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax1.axvline(1, color='r', linestyle='--', alpha=0.5)
    ax1.grid(True)
    ax1.set_title('B-spline')
    
    ax2.plot(theta, dy, 'g-', linewidth=2)
    ax2.axvline(0, color='r', linestyle='--', alpha=0.5)
    ax2.axvline(1, color='r', linestyle='--', alpha=0.5)
    ax2.grid(True)
    ax2.set_title('Derivative')
    
    plt.tight_layout()
    plt.show()

def test_casadi_bspline():
    # Step 1: Initialize control points
    q0 = np.array([0, 5, 10, 15])
    q1 = np.array([1, 6, 11, 16])
    c = np.linspace(q0, q1, 3)  # 7 control points from np.zeros(7) to np.ones(7)
    
    # Step 2: Generate the knot vector
    n_control_points = len(c)  # Number of control points
    k = 2  # Spline order
    t = knot_vector(n_control_points, k)

    # Step 3: Evaluate the B-spline using CasADi
    theta = np.linspace(-0.2, 1.2, 100)  # Evaluation points
    y = np.array([casadiBspline(x, t, c, k) for x in theta])

    print("t: ", t)
    print("c: ", c)
    print("y: ", y)

    # Step 4: Plot the B-spline
    plt.plot(theta, y.squeeze())
    plt.grid()
    plt.show()


def test_numpy_bspline():
  n_ctrl_pts = 3
  k = 2
  q_start = np.ones((9,))*0.5
  q_end = np.ones((9,))*0.4
  via_pts = (np.linspace(q_start, q_end, n_ctrl_pts))
  # compute bspline
  ctr_pts, knot_vec = compute_control_points(via_pts, k)
  np.set_printoptions(precision=2, suppress=True)
  print("via_pts: \n", via_pts)
  print("ctr pts: \n", ctr_pts)
  # ctr_pts = via_pts

  u_vec = np.linspace(0, 1, 100)
  q_vec = np.array([bspline(u, knot_vec, ctr_pts, k) for u in u_vec])

  plt.figure()
  for i in range(9):
      plt.plot(u_vec, q_vec[:, i], label=f'q_vec[:, {i}]')
  plt.grid()
  plt.legend()
  plt.show()


def test_scipy_bspline():
    import numpy as np
    from scipy.interpolate import BSpline
    import matplotlib.pyplot as plt

    n_ctrl_pts = 3
    k = 3
    q_start = np.ones((9,))*0.5
    q_end = np.ones((9,))*0.4
    via_pts = (np.linspace(q_start, q_end, n_ctrl_pts))
    # compute bspline
    ctr_pts, knot_vec = compute_control_points(via_pts, k)

    print("ctr_pts: \n", ctr_pts)
    print("knot_vec: \n", knot_vec)

    # Create the B-spline object
    bspline = BSpline(knot_vec, ctr_pts.T, k)

    # Evaluate the B-spline at the given points
    points = np.linspace(0, 3, 100)
    values = bspline(points)

    # Plot the B-spline
    plt.plot(ctr_pts[:, 0], ctr_pts[:, 1], 'ro-', label='Control Points')
    plt.plot(values[:, 0], values[:, 1], 'b-', label='B-Spline')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('B-Spline Interpolation')
    plt.legend()
    plt.grid()
    plt.show()





if __name__ == '__main__':
    # test_bspline()
    # test_casadi_bspline()
    test_bspline_simple()

    # test_numpy_bspline()

    # test_scipy_bspline()