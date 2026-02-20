import numpy as np
import matplotlib.pyplot as plt

class FirstOrderIIR:
    def __init__(self, u_init, y_init, Ta):
        self.Ta = Ta
        self.a = np.ones(1)
        self.b = np.ones(2)
        self.u_m = u_init
        self.y_m = y_init

    def update(self, u):
        y = (self.b[0] * u + self.b[1] * self.u_m - self.a * self.y_m)
        self.u_m = u
        self.y_m = y
        return y

    def reset(self, u_reset, y_reset):
        self.u_m = u_reset
        self.y_m = y_reset

    def get_time_constant(self, f_c_norm):
        f_min_N = 1e-6 * np.ones(1)
        f_c_N = np.minimum(np.ones(1), np.maximum(f_min_N, f_c_norm))
        f_max = 1.0 / (2.0 * self.Ta)
        T_1 = 1.0 / (2.0 * np.pi * f_c_N * f_max)
        return T_1

class DT1(FirstOrderIIR):
    def __init__(self, u_init=0.0, y_init=0.0, gain=1.0, f_c_norm=0.5, Ta=1.0):
        super().__init__(u_init, y_init, Ta)
        self.T_R = self.get_time_constant(f_c_norm)
        self.V_D = gain * self.T_R / Ta * (np.ones(1) - np.exp(-Ta / self.T_R))
        
        self.b[0] = self.V_D / self.T_R
        self.b[1] = -self.b[0]
        self.a = -np.exp(-Ta / self.T_R)

class PT1(FirstOrderIIR):
    def __init__(self, u_init=0.0, y_init=0.0, gain=1.0, f_c_norm=0.5, Ta=1.0):
        super().__init__(u_init, y_init, Ta)
        self.V = gain
        self.T_1 = self.get_time_constant(f_c_norm)
        tmp = np.exp(Ta / self.T_1)
        
        self.b[0] = 0.0
        self.b[1] = (tmp - 1.0) * self.V / tmp
        self.a = -1.0 / tmp

def main():
    Ta = 0.01
    time_steps = 100
    input_signal = np.concatenate([np.zeros(50), np.ones(time_steps-50)])
    
    dt1_filter = DT1(Ta=Ta, f_c_norm=0.05)
    pt1_filter = PT1(Ta=Ta, f_c_norm=0.05)
    
    dt1_output = []
    pt1_output = []
    
    for u in input_signal:
        dt1_output.append(dt1_filter.update(u))
        pt1_output.append(pt1_filter.update(u))
    
    plt.figure()
    plt.plot(dt1_output, label="DT1 Output")
    plt.plot(pt1_output, label="PT1 Output")
    plt.legend()
    plt.xlabel("Time Step")
    plt.ylabel("Output")
    plt.title("Filter Response to Unit Step Input")
    plt.show()

if __name__ == "__main__":
    main()