from numpy.typing import NDArray
import numpy as np


class Utils:

    @staticmethod
    def homogenize_matrix(mat: NDArray):
        """
        mat is M X N
        return (M+1) X N with the last row of all 1s
        """
        return np.vstack((mat, np.ones(mat.shape[1])))

    @staticmethod
    def dehomogenize_matrix(mat: NDArray):
        """
        mat is M x N
        return (M-1) X N where each of the M-1 Rows are divided by the Mth column
        """
        return mat[:-1, :] / mat[-1, :]

    @staticmethod
    def homogenous_mat_mult(A: NDArray, B: NDArray):
        """
        Perform A @ [B, 1] (homogenize) and then dehomogenize
        """
        return Utils.dehomogenize_matrix(A @ Utils.homogenize_matrix(B))
