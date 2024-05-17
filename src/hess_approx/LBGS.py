from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import scipy
import torch
from scipy import sparse
from numpy import linalg as LA
from numpy.linalg import inv
from scipy.linalg import eigh
from scipy.linalg import eig, eigh
from scipy.sparse.linalg import eigs, eigsh
from abc import ABCMeta, abstractmethod


class LBFGS():
    def __init__(self):
        self.memory_init = False
        self.Y = None
        self.S = None
        self.M = None
        self.Psi = None
        self.gamma = 1.0

        self.config = dict(
            ha_type="LBGS",
            ha_memory=3,
            ha_eig_type="standard",
            ha_tol=1e-15,
            ha_r=1e-8,
            ha_print_errors=False,
            ha_update_type="overlap",  # "sampling"
            ha_sampling_radius=0.05,
        )

        self.memory = self.config.get("ha_memory")
        self.tol = self.config.get("ha_tol")
        self.r = self.config.get("ha_r")
        self.eig_type = self.config.get("ha_eig_type")
        self.print_errors = self.config.get("ha_print_errors")
        self.sampling_radius = self.config.get("ha_sampling_radius")


    def update_memory(self, s, y):
        perform_up = True
        if (np.dot(s, y) < 1e-16):
            perform_up = False
            print("LBFGS:: Skipping update ... ")

        if (perform_up == True):
            if (self.memory_init == False):
                self.S = s.reshape(len(s), 1)
                self.Y = y.reshape(len(y), 1)
                self.memory_init = True
            elif (self.S.shape[1] < self.memory):
                self.S = np.c_[self.S, s]
                self.Y = np.c_[self.Y, y]
            else:
                # add new, remove oldest
                self.S[:, 0] = s
                self.Y[:, 0] = y
                self.S = np.roll(self.S, -1, axis=1)
                self.Y = np.roll(self.Y, -1, axis=1)

            # loop until there is something to be removed
            while (self.precompute(s, y) and self.S.shape[1] > 0):
                # Delete column at index 0
                self.S = np.delete(self.S, 0, axis=1)
                self.Y = np.delete(self.Y, 0, axis=1)

            if (self.S.shape[1] == 0):
                self.memory_init == False

        return self.S, self.Y


    def update_memory_inv(self, s, y, gamma=None):
        perform_up = True
        if (np.dot(s, y) < 1e-16):
            perform_up = False
            print("LBFGS:: Skipping update ... ")

        if (perform_up == True):
            if (self.memory_init == False):
                self.S = s.reshape(len(s), 1)
                self.Y = y.reshape(len(y), 1)
                self.memory_init = True
            elif (self.S.shape[1] < self.memory):
                self.S = np.c_[self.S, s]
                self.Y = np.c_[self.Y, y]
            else:
                # add new, remove oldest
                self.S[:, 0] = s
                self.Y[:, 0] = y
                self.S = np.roll(self.S, -1, axis=1)
                self.Y = np.roll(self.Y, -1, axis=1)

            # loop until there is something to be removed
            while (self.precompute_inv(s, y, gamma) and self.S.shape[1] > 0):
                # Delete column at index 0
                self.S = np.delete(self.S, 0, axis=1)
                self.Y = np.delete(self.Y, 0, axis=1)

            if (self.S.shape[1] == 0):
                self.memory_init == False

        return perform_up


    def precompute_inv(self, s=None, y=None, gamma=None):
        SY = np.matmul(np.transpose(self.S), self.Y)
        D = np.diag(np.diag(SY))
        L = np.tril(SY, k=-1)
        # R = SY - L
        R = np.triu(SY)
        DLL = (D + L + np.transpose(L))

        if (gamma == None):
            self.gamma = self.init_eig_min_inv(DLL, s, y)
        else:
            self.gamma = gamma

        Ygamma = self.gamma * self.Y
        YY = np.matmul(np.transpose(self.Y), self.Y)

        R_inv = inv(R)
        R_inv_T = np.transpose(R_inv)

        term11 = R_inv_T*(D + self.gamma * YY)*R_inv
        term22 = np.zeros((self.S.shape[1], self.S.shape[1]))

        M_inv_1 = np.c_[term11, -R_inv_T]
        M_inv_2 = np.c_[-R_inv, term22]
        self.M_inv = np.r_[M_inv_1, M_inv_2]

        self.Psi = np.c_[self.S, Ygamma]

        return False


    def precompute(self, s=None, y=None):
        if s is None or y is None:
            s = self.S[:, -1]
            y = self.Y[:, -1]

        SY = np.matmul(np.transpose(self.S), self.Y)
        D = np.diag(np.diag(SY))
        L = np.tril(SY, k=-1)
        DLL = (D + L + np.transpose(L))

        self.gamma = self.init_eig_min(DLL, s, y)

        Sgamma = self.gamma * self.S
        SS = np.matmul(np.transpose(self.S), self.S)

        term11 = -self.gamma * SS

        M_inv_1 = np.c_[term11, -L]
        M_inv_2 = np.c_[-np.transpose(L), D]
        self.M_inv = np.r_[M_inv_1, M_inv_2]

        # self.M      = inv(self.M_inv)
        self.Psi = np.c_[Sgamma, self.Y]
        PsiPsi = np.matmul(np.transpose(self.Psi), self.Psi)

        try:
            self.M = inv(self.M_inv)
        except np.linalg.LinAlgError:
            if (self.print_errors):
                print("Problem in update M ---- ")
            return True

        try:
            # todo:: store R, so we have it
            R = scipy.linalg.cholesky(PsiPsi, lower=False)
        except scipy.linalg.LinAlgError:
            if (self.print_errors):
                print("Problem in update for psi ---- ")
            return True

        try:
            helpp = np.linalg.solve(self.M_inv, np.transpose(R))
        except np.linalg.LinAlgError:
            if (self.print_errors):
                print("Problem in update M_inv ---- ")
            return True

        return False


    def apply(self, v):
        if (self.memory_init == False):
            return self.gamma * v

        result = np.matmul(np.transpose(self.Psi), v)
        result = np.matmul(self.M, result)
        result = np.matmul(self.Psi, result)
        result = (self.gamma*v) + result
        return result


    def apply_inv(self, v):
        if (self.memory_init == False):
            return self.gamma * v

        result = np.matmul(np.transpose(self.Psi), v)
        result = np.matmul(self.M_inv, result)
        result = np.matmul(self.Psi, result)
        result = (self.gamma*v) + result
        return result


    def init_eig_min_inv(self, DLL, s, y):
        if (self.eig_type == "one"):
            return 1.0
        elif (self.eig_type == "standard"):
            gamma = np.dot(s, y)
            gamma = gamma/np.dot(y, y)

            if (np.absolute(gamma) < self.tol):
                gamma = 1.0

            return gamma
        else:
            print("------------- not defined type of init_eig_min ---------")
            exit(0)


    def init_eig_min(self, DLL, s, y):

        if (self.eig_type == "one"):
            return 1.0
        elif (self.eig_type == "standard"):
            gamma = np.dot(s, y)
            gamma = np.dot(y, y)/gamma

            if (np.absolute(gamma) < self.tol):
                gamma = 1.0

            return gamma

        elif (self.eig_type == "eigen_decomp"):
            SS = np.matmul(np.transpose(self.S), self.S)
            eig_min = 1.0

            try:
                eigvals = eigh(DLL, SS, eigvals_only=True, check_finite=False)
                eig_min = np.amin(eigvals)
            except scipy.linalg.LinAlgError:
                eig_min = 1.0

            if (eig_min <= 0):
                gamma = np.dot(y, y)/np.dot(s, y)

                if (np.absolute(gamma) < self.tol):
                    gamma = 1.0
                return gamma

            else:
                eig_min = 0.9*eig_min

            if (np.absolute(eig_min) < self.tol):
                eig_min = 1.0

            return eig_min  # for inverse approx

        else:
            print("------------- not defined type of init_eig_min ---------")
            exit(0)


    def reset_memory(self):
        self.memory_init = False
        self.S = []
        self.Y = []
        self.M_inv = []
        self.M = []
        self.gamma = 1.0
        self.Psi = []
