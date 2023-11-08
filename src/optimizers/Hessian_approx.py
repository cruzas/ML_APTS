from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import os
import matplotlib.pyplot as plt
import time

import scipy
from scipy import sparse
from numpy import linalg as LA
from numpy.linalg import inv
from scipy.linalg import eigh
from numpy.linalg import inv
from scipy import array, linalg, dot

from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import scipy.linalg
from scipy.linalg import eigh
from scipy.linalg import eig


class OBS_TORCH():
    def __init__(self):
        super(OBS_TORCH, self).__init__()
        self.tol = 1e-5

    def solve_tr_subproblem(self, grads, delta, gamma, Psi, M_inv):
        assert not torch.isnan(grads).any()
        # assert not torch.isnan(delta).any()
        assert not torch.isnan(gamma).any()
        assert not torch.isnan(Psi).any()
        assert not torch.isnan(M_inv).any()

        PsiPsi = torch.matmul(Psi.transpose(0, 1), Psi)
        R = torch.linalg.cholesky(PsiPsi, upper=True)

        help111 = torch.linalg.solve(M_inv, R.transpose(0, 1))
        RMR = torch.matmul(R, help111)
        RMR = (RMR + RMR.transpose(0, 1)) / 2.0 # this forces eigvenvalues to be real

        D, U = torch.linalg.eig(RMR)
        D = torch.real(D) # eigenvalues are real
        sorted_indices = torch.argsort(D)
        D = D[sorted_indices]
        U = U[:, sorted_indices]

        sizeD = D.shape[0]
        Lambda_one = D + gamma
        Lambda = torch.cat((Lambda_one, gamma))
        Lambda[torch.abs(Lambda) < self.tol] = 0
        lambda_min = torch.min(Lambda[0], gamma)

        if torch.sum(torch.abs(torch.imag(U)))>10**-10:
            raise ValueError('remove forcing real dtype')
        U = torch.real(U)
        RU = torch.linalg.solve(R, U)

        P_parallel = torch.matmul(Psi, RU)
        Psig = torch.matmul(Psi.transpose(0, 1), grads)
        g_parallel = torch.matmul(RU.transpose(0, 1), Psig)

        gg = grads@grads#torch.matmul(grads.transpose(0, 1), grads)
        gpgp = g_parallel@g_parallel#torch.matmul(g_parallel.transpose(0, 1), g_parallel)

        a_kp2 = torch.sqrt(torch.abs(gg - gpgp))
        if a_kp2**2 < self.tol:
            a_kp2 = torch.tensor(0.0)

        a_j = torch.cat((g_parallel, a_kp2.view(-1)))
        helpp = a_j / Lambda

        if lambda_min > 0 and torch.norm(helpp) <= delta:
            pStar = self.ComputeSBySMW(gamma, grads, Psig, Psi, M_inv, PsiPsi)
            return pStar
        elif lambda_min <= 0 and self.phiBar_f(-lambda_min, Lambda, a_j, delta) > 0:
            sigmaStar = -lambda_min
            v = torch.zeros(sizeD + 1)
            idx_pseudo = torch.where(torch.abs(Lambda + sigmaStar) > self.tol)
            v[idx_pseudo] = a_j[idx_pseudo] / (Lambda[idx_pseudo] + sigmaStar)

            if torch.abs(gamma + sigmaStar) < self.tol:
                pStar = -1.0 * torch.matmul(P_parallel, v[:sizeD])
            else:
                term1 = -1.0 * torch.matmul(P_parallel, v[:sizeD])
                term_help = torch.linalg.solve(PsiPsi, Psig)
                term2 = 1.0 / (gamma + sigmaStar) * torch.matmul(Psi, term_help)
                term3 = grads / (gamma + sigmaStar)
                pStar = term1 + term2 - term3

            if lambda_min < 0:
                alpha = torch.sqrt(delta**2 - torch.matmul(pStar.transpose(0, 1), pStar))
                pHatStar = pStar

                if torch.abs(lambda_min - Lambda[0]) < self.tol:
                    zstar = (1.0 / torch.norm(P_parallel[:, 0])) * alpha * P_parallel[:, 0]
                else:
                    e = torch.zeros_like(grads)
                    found = False

                    for i in range(sizeD):
                        e[i] = 1
                        u_min = e - torch.matmul(P_parallel, P_parallel[i, :].transpose(0, 1))
                        if torch.norm(u_min) > self.tol:
                            found = True
                            break

                        e[i] = 0

                    if not found:
                        e[sizeD] = 1
                        u_min = e - torch.matmul(P_parallel, P_parallel[sizeD, :].transpose(0, 1))

                    u_min = u_min / torch.norm(u_min)
                    zstar = alpha * u_min

                pStar = pHatStar + zstar

            return pStar
        else:
            if lambda_min > 0:
                sigmaStar = self.Newton(0, Lambda, a_j, delta)
            else:
                sigmaHat = torch.max(a_j / delta - Lambda)
                if sigmaHat > -lambda_min:
                    sigmaStar = self.Newton(sigmaHat, Lambda, a_j, delta)
                else:
                    sigmaStar = self.Newton(-lambda_min, Lambda, a_j, delta)

            if torch.isnan(sigmaStar) or torch.isinf(sigmaStar):
                sigmaStar = self.Newton(0, Lambda, a_j, delta)

            pStar = self.ComputeSBySMW(gamma + sigmaStar, grads, Psig, Psi, M_inv, PsiPsi)
            return pStar

    def ComputeSBySMW(self, tauStar, g, Psig, Psi, invM, PsiPsi):
        ww = (tauStar**2 * invM) + tauStar * PsiPsi
        term1 = torch.linalg.solve(ww, Psig)
        p_star = (-torch.tensor(1.0) / tauStar * g) + torch.matmul(Psi, term1)
        return p_star

    def phiBar_f(self, sigma, Dd, a_j, delta):
        m = a_j.shape[0]
        D = Dd + sigma

        test1 = torch.zeros(m)
        test2 = torch.zeros(m)

        test1[torch.abs(a_j) < self.tol] = 1
        test2[torch.abs(D) < self.tol] = 1

        t1 = torch.sum(test1)
        t2 = torch.sum(test2)

        if t1 > 0 or t2 > 0:
            phiBar = -1 / delta
            return phiBar

        pnorm2 = 0
        for i in range(m):
            if torch.abs(a_j[i]) > self.tol and torch.abs(D[i]) > self.tol:
                pnorm2 = pnorm2 + (a_j[i] / D[i])**2

        normP = torch.sqrt(pnorm2)
        phiBar = 1.0 / normP - 1.0 / delta
        return phiBar

    def Newton(self, x0, Lambda, a_j, delta):
        maxIter = 200

        x = x0
        k = 0

        f, g = self.phiBar_fg(x, Lambda, a_j, delta)

        while torch.abs(f) > self.tol and k < maxIter:
            x = x - f / g
            f, g = self.phiBar_fg(x, Lambda, a_j, delta)
            k = k + 1

        if torch.isnan(x) or torch.isinf(x):
            print('asd')

        return x

    def phiBar_fg(self, sigma, Dd, a_j, delta):
        m = a_j.shape[0]
        D = Dd + sigma
        phiBar_g = 0

        test1 = torch.zeros(m)
        test2 = torch.zeros(m)

        test1[torch.abs(a_j) < self.tol] = 1
        test2[torch.abs(D) < self.tol] = 1

        t1 = torch.sum(test1)
        t2 = torch.sum(test2)

        if t1 > 0 or t2 > 0:
            phiBar = torch.tensor(-1) / delta
            phiBar_g = torch.tensor(1) / self.tol
            return phiBar, phiBar_g


        p = a_j / D
        normP = torch.norm(p)
        phiBar = 1 / normP - 1 / delta

        phiBar_g = torch.sum((a_j**2) / (D**3))
        phiBar_g = phiBar_g / (normP**3)

        return phiBar, phiBar_g




class OBS():
  def __init__(self):
    super(OBS, self).__init__()
    self.tol  = 10**-5


  def solve_tr_subproblem(self, grads, delta, gamma, Psi, M_inv):
    assert np.isnan(grads).any() == False
    assert np.isnan(delta) == False
    assert np.isnan(gamma) == False
    assert np.isnan(Psi).any() == False
    assert np.isnan(M_inv).any() == False

    PsiPsi = np.matmul(np.transpose(Psi), Psi)
    R = scipy.linalg.cholesky(PsiPsi, lower=False)  # should return upper triangular 

    help111 = np.linalg.solve(M_inv, np.transpose(R))
    RMR = np.matmul(R, help111)
    # RMR = np.matmul(R, np.matmul(self.M, np.transpose(R)))
    RMR = (RMR + np.transpose(RMR))/2.0

    D, U = LA.eig(RMR)

    idx = D.argsort()
    D = D[idx]
    U = U[:,idx]

    sizeD       = D.shape[0]
    Lambda_one  = D + gamma
    Lambda = np.append(Lambda_one, gamma)
    Lambda[np.absolute(Lambda) < self.tol] = 0
    lambda_min  = np.minimum(Lambda[0], gamma)


    # RU = inv(R)*U  # inv sometimes fails, so solve instead  
    RU = np.linalg.solve(R, U)
    
    P_parallel  = np.matmul(Psi, RU) # Psi = Y - B_0 S = QR, but R from line 51 does not seem to be the same R? Also, P_parallel = QU in the paper. 
    Psig        = np.matmul(np.transpose(Psi), grads) 
    g_parallel  = np.matmul(np.transpose(RU), Psig) # In the paper, g_parallel = P_parallel^T * grads. 

    gg = np.matmul(np.transpose(grads),grads)
    gpgp = np.matmul(np.transpose(g_parallel),g_parallel)

    a_kp2 = np.sqrt(np.absolute(gg - gpgp))
    if(a_kp2**2 < self.tol):
        a_kp2 = 0

    a_j = np.append(g_parallel, a_kp2)
    helpp =  np.transpose(a_j)/Lambda


    if(lambda_min>0 and (np.linalg.norm(helpp)<=delta)):
        pStar = self.ComputeSBySMW(gamma, grads, Psig, Psi, M_inv, PsiPsi)
        return pStar
    elif((lambda_min<=0) and (self.phiBar_f(-lambda_min, Lambda, a_j, delta) >0)):

        sigmaStar = -lambda_min
        v = 0.0*np.append(D, 0.0)
        idx_pseudo = np.where(np.absolute(Lambda + sigmaStar) > self.tol)
        v[idx_pseudo] = a_j[idx_pseudo]/(Lambda[idx_pseudo]+sigmaStar)

        if(np.absolute(gamma + sigmaStar) < self.tol):
            pStar = -1.0*np.matmul(P_parallel,v[0:sizeD])
        else:
            term1 = -1.0*np.matmul(P_parallel, v[0:sizeD])
            # term_help = np.linalg.solve(PsiPsi, Psig)
            try: 
                term_help = np.linalg.solve(PsiPsi, Psig)
            except np.linalg.LinAlgError:
                try: 
                    term_help = np.linalg.solve(PsiPsi+1e-12, Psig)
                except np.linalg.LinAlgError:
                    try: 
                        term_help = np.linalg.solve(PsiPsi+1e-9, Psig)
                    except np.linalg.LinAlgError:
                        try: 
                            term_help = np.linalg.solve(PsiPsi+1e-6, Psig)
                        except np.linalg.LinAlgError:
                            print("----- PsiPsi::Psig 4  ")
                            exit(0)

            # term2 = 1./(self.gamma+sigmaStar)*(np.matmul(self.Psi, np.matmul(inv(PsiPsi), Psig)))
            term2 = 1./(gamma+sigmaStar)*(np.matmul(Psi, term_help))
            term3 = (grads/(gamma+sigmaStar)) 
            pStar = term1 + term2 - term3
        
        if(lambda_min < 0):
            alpha = np.sqrt(delta**2 - (np.matmul(np.transpose(pStar), pStar))) # Is it ensured that pStart exists before this line?
            if np.isnan(alpha): #NOTE: This is a quick fix proposed by KEN to avoid errors.
                alpha = 10**-5
                
            pHatStar = pStar

            if(np.absolute(lambda_min-Lambda[0])<self.tol):
                zstar = (1./np.linalg.norm(P_parallel[:,0]))*alpha*P_parallel[:,0]
            else:
                e = 0.0*grads
                found=False

                for i in range(0, sizeD):
                    e[i]=1
                    u_min = e - np.matmul(P_parallel, np.transpose(P_parallel[i,:]))
                    if np.linalg.norm(u_min)>self.tol:
                        found = True
                        break

                    e[i]=0

                if(found == False):
                    e[sizeD] = 1
                    u_min = e - np.matmul(P_parallel, np.transpose(P_parallel[sizeD,:]))

                u_min = u_min/(np.linalg.norm(u_min))
                zstar = alpha*u_min

            pStar = pHatStar+zstar 

            assert np.isnan(pStar).any() == False
            return pStar
        else:
            assert np.isnan(pStar).any() == False
            return pStar
            
    else:
        if(lambda_min > 0):
            sigmaStar = self.Newton(0, Lambda, a_j, delta)
        else:
            sigmaHat = np.max(a_j/delta - Lambda)
            if (sigmaHat > -lambda_min): 
              sigmaStar = self.Newton(sigmaHat, Lambda, a_j, delta)
            else:
              sigmaStar = self.Newton(-lambda_min, Lambda, a_j, delta)

        if np.isnan(sigmaStar) or np.isinf(sigmaStar):
            sigmaStar = self.Newton(0, Lambda, a_j, delta)

        pStar = self.ComputeSBySMW(gamma+sigmaStar, grads, Psig, Psi, M_inv, PsiPsi)
        assert np.isnan(pStar).any() == False
        return pStar


  def ComputeSBySMW(self, tauStar, g, Psig, Psi, invM, PsiPsi):
    ww = (tauStar*tauStar*invM) + tauStar*PsiPsi
    # p_star  = (-1./tauStar*g) + np.matmul(Psi, np.matmul(inv(ww), Psig))
    # more stable than inv 
    try:
        term1 = np.linalg.solve(ww, Psig)
    except np.linalg.LinAlgError:
        # print("----- here, catching errorr ")
        try: 
            term1 = np.linalg.solve(ww+1e-12, Psig)
        except np.linalg.LinAlgError:
            try: 
                term1 = np.linalg.solve(ww+1e-9, Psig)
            except np.linalg.LinAlgError:
                try: 
                    term1 = np.linalg.solve(ww+1e-6, Psig)
                except np.linalg.LinAlgError:
                    try: 
                        term1 = np.linalg.solve(ww+1e-3, Psig)
                    except np.linalg.LinAlgError:
                        try: 
                            term1 = np.linalg.solve(ww+1.0, Psig)
                        except np.linalg.LinAlgError:
                            print("----- ComputeSBySMW::fail 6  ")
                            raise Exception("----- ComputeSBySMW::fail 6  ")



    p_star =  (-1./tauStar*g) + np.matmul(Psi, term1)
    assert np.isnan(p_star).any() == False
    return p_star



  def phiBar_f(self, sigma, Dd, a_j, delta):
    m = a_j.shape[0]
    D = Dd + sigma

    test1 = 0*a_j
    test2 = 0*D

    test1[np.absolute(a_j) < self.tol] = 1
    test2[np.absolute(D) < self.tol] = 1

    t1 = np.sum(test1)
    t2 = np.sum(test2)


    # we found zero, so do safeguards
    if(t1 > 0 or t2 > 0): 
        pnorm2 = 0;
        for i in range(0, m):
            if((np.absolute(a_j[i]) > self.tol) and (np.absolute(D[i]) < self.tol)):
                phiBar = -1/delta
                return phiBar
            elif((np.absolute(a_j[i]) > self.tol) and (np.absolute(D[i]) > self.tol)):
                pnorm2 = pnorm2 + (a_j[i]/D[i])**2
            else: # this is made up by KEN to avoid errors. Sometimes it happens that none of the above conditions are met.
                pnorm2 = pnorm2 + self.tol

        phiBar = np.sqrt(1./pnorm2) - 1./delta
        return phiBar

    p = a_j/D
    phiBar = 1./np.linalg.norm(p) - 1./delta
    return phiBar




  def Newton(self, x0, Lambda, a_j, delta):
    maxIter = 200

    x = x0
    k = 0

    f, g  = self.phiBar_fg(x, Lambda, a_j, delta)

    while((np.absolute(f) > self.tol) and  (k < maxIter)):
        x = x - f/g
        f, g  = self.phiBar_fg(x, Lambda, a_j, delta)        
        k = k + 1

    if np.isnan(x) or np.isinf(x):
        print('asd')
    return x




  def phiBar_fg(self, sigma, Dd, a_j, delta):
    m = a_j.shape[0]
    D = Dd + sigma
    phiBar_g = 0

    test1 = 0*a_j
    test2 = 0*D

    test1[np.absolute(a_j) < self.tol] = 1
    test2[np.absolute(D) < self.tol] = 1

    t1 = np.sum(test1)
    t2 = np.sum(test2)

    # we found zero, so let's do safeguards
    if(t1 > 0 or t2 > 0): 
        phiBar = -1/delta
        phiBar_g = 1./self.tol
        return phiBar, phiBar_g
        #NOTE: KEN modified this line to avoids errors. Sometimes it happens that none of the below conditions are met.

        # pnorm2 = 0
        # for i in range(0, m):
        #     if((np.absolute(a_j[i]) > self.tol) and (np.absolute(D[i]) < self.tol)): 
        #         phiBar = -1/delta
        #         phiBar_g = 1./self.tol
        #         return phiBar, phiBar_g
        #     elif((np.absolute(a_j[i]) > self.tol) and (np.absolute(D[i]) > self.tol)):
        #         pnorm2 = pnorm2 + (a_j[i]/D[i])**2
        #         phiBar_g = phiBar_g + ((a_j[i])**2)/((D[i])**3)

        # normP       = np.sqrt(pnorm2)
        # phiBar      = 1./normP - 1./delta
        # assert normP!=0, "normP is zero"
        # phiBar_g    = phiBar_g/(normP**3)
        # assert phiBar_g!=0, "phiBar_g is zero"
        # return phiBar, phiBar_g

    p = np.transpose(a_j)/D
    normP  = np.linalg.norm(p)
    phiBar = 1./normP - 1./delta

    phiBar_g = np.sum((a_j**2)/(D**3))
    phiBar_g = phiBar_g/(normP**3)
    assert phiBar_g!=0, "phiBar_g is zero"
    return phiBar, phiBar_g
