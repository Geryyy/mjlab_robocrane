import numpy as np
from sspp import BSplines as bs
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt



def closest_path_parameter(x_ref, knot_vec, ctr_pts, k):
    """
    Finds the B-spline parameter theta that yields the closest point on the spline to x_ref.
    
    Parameters:
        x_ref (np.array): Reference point to find the closest spline point to.
        knot_vec (np.array): Knot vector for the B-spline.
        ctr_pts (np.array): Control points of the B-spline.
        k (int): Degree of the B-spline.
    
    Returns:
        float: The optimal spline parameter theta.
    """
    # theta = 0.0
    # x = bs.bspline(theta, knot_vec, ctr_pts, k)
    # x_deriv = bs.bspline_derivative(theta, knot_vec, ctr_pts, k)


# Define the objective function: squared distance between x_ref and the spline point
    def objective(theta):
        x_spline = bs.bspline(theta, knot_vec, ctr_pts, k)
        return np.linalg.norm(x_spline - x_ref)**2  # Squared Euclidean distance
    
    # Optimize theta in the valid range (typically [0, 1] for normalized B-splines)
    result = minimize_scalar(objective, bounds=(0.0, 1.0), method='bounded')
    
    return result.x  # Return the optimal theta
    



def main():
    # Define control points and knot vector
    n_ctrl_pts = 5
    k = 3
    knot_vec = bs.knot_vector(n_ctrl_pts, k)
    ctr_pts = np.array([[0, 0], [1, 2], [3, 3], [4, 1], [5, 0]])
    
    # Define a reference point (not on the spline)
    x_ref = np.array([2.5, 1.5])
    
    # Find the closest parameter
    theta_opt = closest_path_parameter(x_ref, knot_vec, ctr_pts, k)
    closest_point = bs.bspline(theta_opt, knot_vec, ctr_pts, k)

    print(f"Optimal theta: {theta_opt}")
    print(f"Closest point on spline: {closest_point}")


    for theta in np.linspace(theta_opt*0.9, theta_opt*1.1, 10):
        x_spline = bs.bspline(theta, knot_vec, ctr_pts, k)
        print(f"theta: {theta}, x_spline: {x_spline} dist: {np.linalg.norm(x_spline - x_ref)}")
    
    # Plot the B-spline curve
    theta_vals = np.linspace(0, 1, 100)
    spline_pts = np.array([bs.bspline(t, knot_vec, ctr_pts, k) for t in theta_vals])
    
    plt.plot(spline_pts[:, 0], spline_pts[:, 1], label='B-spline Curve', color='blue')
    plt.scatter(ctr_pts[:, 0], ctr_pts[:, 1], color='red', label='Control Points')
    plt.scatter(*x_ref, color='green', marker='x', s=100, label='Reference Point')
    plt.scatter(*closest_point, color='purple', marker='o', s=100, label='Closest Point on Spline')
    plt.legend()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Closest Point on B-spline to Reference Point')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
