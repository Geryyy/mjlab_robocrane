import numpy as np


def quat_normalize(q):
    return q / np.linalg.norm(q)

def quat_mul(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_integrate(q, w, dt):
    # qdot = 0.5 * (0, w) ⊗ q
    wq,xq,yq,zq = q
    wx,wy,wz = w
    qdot = 0.5 * np.array([
        -xq*wx - yq*wy - zq*wz,
         wq*wx + yq*wz - zq*wy,
         wq*wy - xq*wz + zq*wx,
         wq*wz + xq*wy - yq*wx
    ])
    return quat_normalize(q + dt * qdot)