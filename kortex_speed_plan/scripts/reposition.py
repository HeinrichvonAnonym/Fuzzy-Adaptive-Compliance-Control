import numpy as np

R_A = np.array([[1, 0, 0], 
             [0, 1, 0],
             [0, 0, 1]])
# A -> World

R_B = np.array([[0, 1, 0], 
             [-1, 0, 0],
             [0, 0, 1]])
# B -> World

d_P = np.array([-2.3, 1.8, 0])
# d_P = np.dot(R_B, d_P)
# print(d_P)
# A -> B

def get_lamdas(th_x1, th_y1, th_x2, th_y2):
    lamda_as = np.zeros(3)
    lamda_bs = np.zeros(3)
    for i in range(3):
        lamda_as[i] = (     R_A[i, 0] * np.cos(th_y1) * np.sin(th_x1) 
                       +    R_A[i, 1] * np.cos(th_y1) * np.cos(th_x1) + 
                            R_A[i, 2] * np.sin(th_y1))
        lamda_bs[i] = (     R_B[i, 0] * np.cos(th_y2) * np.sin(th_x2) 
                       +    R_B[i, 1] * np.cos(th_y2) * np.cos(th_x2) + 
                            R_B[i, 2] * np.sin(th_y2))
    return lamda_as, lamda_bs

def get_ds(lamda_as, lamda_bs):
    M = np.array([[lamda_as[0], - lamda_bs[0]],
                  [lamda_as[1], - lamda_bs[1]],
                  [lamda_as[2], - lamda_bs[2]]])
    y = np.array([d_P[0], d_P[1], d_P[2]])
    M = np.linalg.pinv(M)
    return np.dot(M, y)

def get_position(ds, th_x1, th_y1, th_x2, th_y2):
    x_a = ds[0] * (R_A[0, 0] * np.cos(th_y1) * np.sin(th_x1) + R_A[0, 1] * np.cos(th_y1) * np.cos(th_x1) + R_A[0, 2] * np.sin(th_y1))
    x_b = ds[1] * (R_B[0, 0] * np.cos(th_y2) * np.sin(th_x2) + R_B[0, 1] * np.cos(th_y2) * np.cos(th_x2) + R_B[0, 2] * np.sin(th_y2)) + d_P[0]
    x = (x_a + x_b) / 2
    d_x = np.abs(x_a - x_b)
    y_a = ds[0] * (R_A[1, 0] * np.cos(th_y1) * np.sin(th_x1) + R_A[1, 1] * np.cos(th_y1) * np.cos(th_x1) + R_A[1, 2] * np.sin(th_y1))
    y_b = ds[1] * (R_B[1, 0] * np.cos(th_y2) * np.sin(th_x2) + R_B[1, 1] * np.cos(th_y2) * np.cos(th_x2) + R_B[1, 2] * np.sin(th_y2)) + d_P[1]
    y = (y_a + y_b) / 2
    d_y = np.abs(y_a - y_b)
    z_a = ds[0] * (R_A[2, 0] * np.cos(th_y1) * np.sin(th_x1) + R_A[2, 1] * np.cos(th_y1) * np.cos(th_x1) + R_A[2, 2] * np.sin(th_y1))
    z_b = ds[1] * (R_B[2, 0] * np.cos(th_y2) * np.sin(th_x2) + R_B[2, 1] * np.cos(th_y2) * np.cos(th_x2) + R_B[2, 2] * np.sin(th_y2)) + d_P[2]
    z = (z_a + z_b) / 2
    d_z = np.abs(z_a - z_b)
    return np.array([x, y, z]), np.array([d_x, d_y, d_z])

def reposition(th_x1, th_y1, th_x2, th_y2):
    lamda_as, lamda_bs = get_lamdas(th_x1, th_y1, th_x2, th_y2)
    ds = get_ds(lamda_as, lamda_bs)
    return get_position(ds, th_x1, th_y1, th_x2, th_y2)

print(reposition(np.arctan(1/8),np.arctan(1/8), np.arctan(2/25), np.arctan(2/25)))