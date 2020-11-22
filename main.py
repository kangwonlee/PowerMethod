from typing import Tuple

import numpy as np
import numpy.linalg as nl


def power_method(matA:np.ndarray, vecX:np.ndarray=None, epsilon:float=1e-7, n_iter_max:int=100000) -> Tuple[float, np.ndarray, int]:
    n = matA.shape[0]
    
    if vecX is None:
        vecX = np.ones(n)

    for i in range(n_iter_max):

        vecY = matA @ vecX
        lam = abs(vecY).max()
        vecY *= 1.0 / lam

        norm = nl.norm(vecX - vecY)
        if norm < epsilon:
            break

        vecX = vecY


    return lam, vecY, i


def main1():
  angle_deg = 30
  angle_rad = np.deg2rad(angle_deg)
  c = np.cos(angle_rad)
  s = np.sin(angle_rad)
  matA = np.array([[c, -s], [s, c]])

  e_val, e_vec, n_iter = power_method(matA)

  print("eigenvalue =", e_val)
  print("eigenvector =", e_vec)
  print("number of iterations =", n_iter)

  print(nl.eig(matA))


def main2():
  angle_deg = 30
  angle_rad = np.deg2rad(angle_deg)
  c = np.cos(angle_rad)
  s = np.sin(angle_rad)
  matA = np.array([[c, -s], [s, c]])

  e_val, e_vec, n_iter = power_method(matA)

  print("eigenvalue =", e_val)
  print("eigenvector =", e_vec)
  print("number of iterations =", n_iter)

  w, v = nl.eig(matA)
  print('w =', w)
  print('v =', v)


if "__main__" == __name__:
  main1()
  main2()
