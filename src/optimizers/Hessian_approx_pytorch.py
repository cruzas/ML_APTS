import torch


class Hessian_approx():
    def __init__(self, gamma, device, memory_length=5, tol=1e-6, method='SR1'):
        super(Hessian_approx, self).__init__()
        self.tol = torch.tensor(tol, device=device)
        self.gamma = torch.tensor(gamma, device=device)
        self.memory_length = memory_length
        self.device = device
        self.S = torch.zeros((0,0), device=device)
        self.Y = torch.zeros((0,0), device=device)
        self.M = torch.zeros((0,0), device=device)
        self.M_inv = torch.zeros((0,0), device=device)
        self.Psi = torch.zeros((0,0), device=device)
        self.method = method

    def update_memory(self, s, y):
        '''
        Update the memory of the Hessian approximation.
        '''
        if len(s.shape) == 1:
            s = s.unsqueeze(1)
        if len(y.shape) == 1:
            y = y.unsqueeze(1)

        if (y - self.B(s)).T @ s != 0:
            if self.S.numel() == 0:
                self.S = s
                self.Y = y
            else:
                self.S = torch.cat((self.S, s), dim=1)
                self.Y = torch.cat((self.Y, y), dim=1)

            if self.S.size(1) > self.memory_length:
                self.S = self.S[:, 1:]
                self.Y = self.Y[:, 1:]

        if self.method == 'SR1':
            self.Psi = self.Y - self.gamma * self.S
            SY = self.S.T @ self.Y
            D = torch.diag(torch.diag(SY))
            L = torch.tril(SY)
            self.M_inv = D + L + L.T - self.S.T @ self.S*self.gamma
            self.M = torch.inverse(self.M_inv)
        else:
            raise NotImplementedError('The method is not implemented yet.')

    def B(self, s):
        if self.S.numel() == 0: # This is the same for all method
            result = s*self.gamma
        elif self.method == 'SR1':
            # Transpose the tensor self.Psi
            a = torch.matmul(self.Psi.T, s)
            # Multiply tensor M with 'a'
            b = torch.matmul(self.M, a)
            # Alternative: If 'self.M_inv' is available, you can use it for solving the equation.
            # b = torch.linalg.solve(self.self.M_inv, a)
            # Calculate the final result
            result = (self.gamma * s) + torch.matmul(self.Psi, b)
        else:
            raise NotImplementedError('The method is not implemented yet.')
        
        return result

    def solve_tr_subproblem(self, grad, radius):
        def check(D):
            if torch.norm(D-D.real, 1)/torch.norm(D, 1)>1e-3:
                print(f'{D} is not real')

        if self.S.numel() == 0:
            return 1.0/self.gamma * grad
        elif self.method == 'SR1':
            # PsiPsi = self.Psi.T.matmul(self.Psi)
            # Q, R = torch.linalg.qr(self.Psi)
            # D, U = torch.linalg.eig(R @ self.M@R.T) # D is the matrix of eigenvalues and U is the matrix with the corresponding eigenvectors as columns
            # idx = torch.argsort(D.real)
            # D = D[idx]
            # U = U[:, idx]

            # sizeD = D.shape[0]
            # R = R + torch.zeros_like(R, device=self.device, dtype=torch.complex64)
            # Lambda_one = D.real + self.gamma
            # Lambda = torch.cat((Lambda_one, torch.tensor([self.gamma], device=self.device)), dim=0)
            # Lambda[torch.abs(Lambda) < self.tol] = 0
            # lambda_min = torch.min(Lambda[0], self.gamma)

            # P_parallel = (Q + torch.zeros_like(Q, device=self.device, dtype=torch.complex64)) @ U
            # Psig = self.Psi.T.matmul(grad)
            # g_parallel = P_parallel.T.matmul(grad + torch.zeros_like(grad, device=self.device, dtype=torch.complex64))

            # gg = grad.T.matmul(grad)
            # gpgp = g_parallel.T.matmul(g_parallel)

            # a_kp2 = torch.sqrt(torch.abs(gg - gpgp))
            # if a_kp2.pow(2) < self.tol:
            #     a_kp2 = 0

            # a_j = torch.cat((g_parallel, torch.tensor([a_kp2], device=self.device)), dim=0)
            # helpp = a_j.T / Lambda
            PsiPsi = self.Psi.T.matmul(self.Psi)
            R = torch.linalg.cholesky(PsiPsi)
            check(R)

            help111 = torch.linalg.solve(self.M_inv, R.T)
            RMR = R.matmul(help111)
            RMR = (RMR + RMR.T)/2.0

            D, U = torch.linalg.eig(RMR)
            idx = torch.argsort(D.real)


            D = D[idx].real
            U = U[:,idx]

            sizeD = D.shape[0]
            Lambda_one  = D + self.gamma
            Lambda = torch.cat((Lambda_one, torch.tensor([self.gamma], device=self.device)), dim=0)
            Lambda[torch.abs(Lambda) < self.tol] = 0
            lambda_min  = torch.min(Lambda[0], self.gamma)

            RU = torch.linalg.solve(R + torch.zeros_like(R, device=self.device, dtype=torch.complex32), U)
            
            P_parallel  = self.Psi.matmul(RU.real)
            Psig        = self.Psi.T.matmul(grad)
            g_parallel  = RU.real.T.matmul(Psig)
            if torch.norm(RU-RU.real)/torch.norm(RU)>1e-3:
                print('RU is not real')

            gg = grad.T.matmul(grad)
            gpgp = g_parallel.T.matmul(g_parallel)

            a_kp2 = torch.sqrt(torch.abs(gg - gpgp))
            if a_kp2.pow(2) < self.tol:
                a_kp2 = 0

            a_j = torch.cat((g_parallel, torch.tensor([a_kp2], device=self.device)), dim=0)
            helpp =  a_j.T/Lambda



            if lambda_min > 0 and (torch.linalg.norm(helpp) <= radius):
                pStar = self.ComputeSBySMW(self.gamma, grad, Psig, PsiPsi)
                return pStar
            elif (lambda_min <= 0) and (self.phiBar_f(-lambda_min, Lambda, a_j, radius) > 0):
                sigmaStar = -lambda_min
                v = torch.zeros_like(Lambda)
                idx_pseudo = torch.where(torch.abs(Lambda + sigmaStar) > self.tol)
                v[idx_pseudo] = a_j[idx_pseudo] / (Lambda[idx_pseudo] + sigmaStar)

                if torch.abs(self.gamma + sigmaStar) < self.tol:
                    pStar = -1.0 * P_parallel.matmul(v[0:sizeD])
                else:
                    term1 = -1.0 * P_parallel.matmul(v[0:sizeD])
                    term_help = torch.linalg.solve(PsiPsi, Psig)
                    term2 = 1.0 / (self.gamma + sigmaStar) * (self.Psi.matmul(term_help))
                    term3 = grad / (self.gamma + sigmaStar)
                    pStar = term1 + term2 - term3

                if lambda_min < 0:
                    alpha = torch.sqrt(radius**2 - (pStar.T.matmul(pStar)))
                    pHatStar = pStar

                    if torch.abs(lambda_min - Lambda[0]) < self.tol:
                        zstar = (
                            (1.0 / torch.linalg.norm(P_parallel[:, 0]))
                            * alpha
                            * P_parallel[:, 0]
                        )
                    else:
                        e = torch.zeros_like(grad)
                        found = False

                        for i in range(0, sizeD):
                            e[i] = 1
                            u_min = e - P_parallel.matmul(
                                P_parallel[i, :].T
                            )
                            if torch.linalg.norm(u_min) > self.tol:
                                found = True
                                break

                            e[i] = 0

                        if found == False:
                            e[sizeD] = 1
                            u_min = e - P_parallel.matmul(
                                P_parallel[sizeD, :].T
                            )

                        u_min = u_min / (torch.linalg.norm(u_min))
                        zstar = alpha * u_min

                    pStar = pHatStar + zstar
                    return pStar
                else:
                    return pStar

            else:
                if lambda_min > 0:
                    sigmaStar = self.Newton(0, Lambda, a_j, radius)
                else:
                    sigmaHat = torch.max(a_j.real / radius - Lambda)
                    if sigmaHat > -lambda_min:
                        sigmaStar = self.Newton(sigmaHat, Lambda, a_j, radius)
                    else:
                        sigmaStar = self.Newton(-lambda_min, Lambda, a_j, radius)

                pStar = self.ComputeSBySMW(
                    self.gamma + sigmaStar, grad, Psig, PsiPsi
                )
                return pStar
        else:
            raise NotImplementedError('The method is not implemented yet.')

    def ComputeSBySMW(self, tauStar, g, Psig, PsiPsi):
        ww = (tauStar * tauStar * self.M_inv) + tauStar * PsiPsi
        term1 = torch.linalg.solve(ww.real, Psig)
        p_star = (-1.0 / tauStar * g) + self.Psi.matmul(term1)
        return p_star

    def phiBar_f(self, sigma, Dd, a_j, radius):
        m = a_j.shape[0]
        D = Dd + sigma
        t1 = torch.sum(
            torch.where(torch.abs(a_j) < self.tol, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        )
        t2 = torch.sum(
            torch.where(torch.abs(D) < self.tol, torch.tensor(1.0, device=self.device), torch.tensor(0.0, device=self.device))
        )

        if t1 > 0 or t2 > 0:
            pnorm2 = 0
            for i in range(0, m):
                if torch.abs(a_j[i]) > self.tol and torch.abs(D[i]) < self.tol:
                    phiBar = -1 / radius
                    return phiBar
                elif torch.abs(a_j[i]) > self.tol and torch.abs(D[i]) > self.tol:
                    pnorm2 = pnorm2 + (a_j[i] / D[i]).pow(2)

            phiBar = torch.sqrt(1.0 / pnorm2) - 1.0 / radius
            return phiBar

        p = a_j.T / D
        phiBar = 1.0 / torch.linalg.norm(p) - 1.0 / radius
        return phiBar

    def Newton(self, x0, Lambda, a_j, radius):
        maxIter = 200

        x = x0
        k = 0

        f, g = self.phiBar_fg(x, Lambda, a_j, torch.tensor(radius, device=self.device))

        while torch.abs(f) > self.tol and k < maxIter:
            x = x - f / g
            f, g = self.phiBar_fg(x, Lambda, a_j, torch.tensor(radius, device=self.device))
            k = k + 1

        return x


    def phiBar_fg(self, sigma, Dd, a_j, radius):
        m = a_j.shape[0]
        D = Dd + sigma
        phiBar_g = torch.tensor(0.0, device=self.device, dtype=torch.float32)

        test1 = torch.zeros_like(a_j).real
        test2 = torch.zeros_like(D).real

        test1[torch.abs(a_j) < self.tol] = 1
        test2[torch.abs(D) < self.tol] = 1

        t1 = torch.sum(test1)
        t2 = torch.sum(test2)

        # we found zero, so let's do safeguards
        if t1 > 0 or t2 > 0:
            pnorm2 = torch.tensor(0.0, device=self.device, dtype=torch.float32)
            for i in range(0, m):
                if (torch.abs(a_j[i]) > self.tol) and (torch.abs(D[i]) < self.tol):
                    phiBar = -1 / radius
                    phiBar_g = 1.0 / self.tol
                    return phiBar, phiBar_g
                elif (torch.abs(a_j[i]) > self.tol) and (torch.abs(D[i]) > self.tol):
                    pnorm2 = pnorm2 + (a_j[i] / D[i]) ** 2
                    phiBar_g = phiBar_g + ((a_j[i]) ** 2) / ((D[i]) ** 3)

            normP = torch.sqrt(pnorm2)
            phiBar = 1.0 / normP - 1.0 / radius
            phiBar_g = phiBar_g / (normP**3)
            return phiBar, phiBar_g

        p = torch.t(a_j) / D
        normP = torch.linalg.norm(p)
        phiBar = 1.0 / normP - 1.0 / radius

        phiBar_g = torch.sum((a_j**2) / (D**3))
        phiBar_g = phiBar_g / (normP**3)
        return phiBar, phiBar_g


# # define a class
# class Hessian_approx():
#     def __init__(self, self.gamma, radius, maxIter, tol, show, device):
#         self.self.gamma = self.gamma
#         self.radius = radius
#         self.maxIter = maxIter
#         self.tol = tol
#         self.show = show
#         self.device = device

#         # History
#         self.memory_length = 10
#         self.S = torch.zeros(0, device=device)
#         self.Y = torch.zeros(0, device=device)


#     def update_memory(self, s, y):
#         '''
#         Update the memory of the Hessian approximation.
#         '''
#         if (y - self.B(s)) @ s != 0:
#             if self.S.numel() == 1:
#                 self.S = s
#                 self.Y = y
#             else:
#                 self.S = torch.cat((self.S, s.unsqueeze(1)), dim=1)
#                 self.Y = torch.cat((self.Y, y.unsqueeze(1)), dim=1)

#             if self.S.size(1) > self.memory_length:
#                 self.S = self.S[:, 1:]
#                 self.Y = self.Y[:, 1:]


#     def obs(self, g, *args):
#         show = 1  # verbosity flag

#         if not ((len(args) == 0 or len(args) == 2)):
#             raise ValueError('Wrong number of inputs...')
#         elif len(args) == 2:
#             self.Psi = args[0]
#             invM = args[1]

#         maxIter = 100  # maximum number of iterations for Newton's method
#         tol = 1e-10  # tolerance for Newton's method

#         if len(args) == 0:  # inputs were S and Y
#             # Compute S'Y, S'S, inv(M), self.Psi
#             SY = torch.matmul(self.S.t(), self.Y)
#             SS = torch.matmul(self.S.t(), self.S)
#             invM = SY + SY.t() - self.self.gamma * SS
#             invM = (invM + invM.t()) / 2  # symmetrize invM, if needed
#             self.Psi = self.Y - self.self.gamma * self.S

#         # Ensure self.Psi is full rank
#         if linalg.matrix_rank(self.Psi) != self.Psi.size(1):
#             raise ValueError('self.Psi is not full rank... exiting obs()')

#         # Compute eigenvalues Hat{Lambda} using Cholesky
#         PsiPsi = torch.matmul(self.Psi.t(), self.Psi)
#         R = torch.cholesky(PsiPsi)
#         RMR = torch.cholesky_solve(torch.eye(self.Psi.size(1)), R)
#         RMR = (RMR + RMR.t()) / 2  # symmetrize RMR', if needed
#         D, U = torch.linalg.eigh(RMR)

#         # Eliminate complex roundoff error then sort eigenvalues and eigenvectors
#         D, ind = torch.sort(D.real)
#         U = U[:, ind]

#         # Compute Lambda as in Equation (7) and lambda_min
#         sizeD = D.size(0)
#         Lambda_one = D + self.gamma * torch.ones(sizeD)
#         Lambda = torch.cat((Lambda_one, self.gamma * torch.ones(1)))
#         Lambda = Lambda * (torch.abs(Lambda) > tol)  # thresholds
#         lambda_min = torch.min(torch.tensor([Lambda[0], self.gamma]))

#         # Define P_parallel and g_parallel
#         RU = torch.cholesky_solve(U.t(), R)
#         P_parallel = torch.matmul(self.Psi, RU)
#         Psig = torch.matmul(self.Psi.t(), g)
#         g_parallel = torch.matmul(RU.t(), Psig)

#         # Compute a_j = (g_parallel)_j for j=1...k+1; a_{k+2}=||g_perp||
#         a_kp2 = torch.sqrt(torch.abs(torch.matmul(g.t(), g) - torch.matmul(g_parallel.t(), g_parallel)))
#         a_kp2 = a_kp2.item() if a_kp2.item()**2 < tol else a_kp2
#         a_j = torch.cat((g_parallel, torch.tensor([a_kp2])))

#         # (1) Check unconstrained minimizer p_u
#         if lambda_min > 0 and torch.norm(a_j / Lambda) <= radius:
#             sigmaStar = torch.tensor(0.0)
#             pStar = ComputeSBySMW(self.gamma, g, Psig, self.Psi, invM, PsiPsi)
#         elif lambda_min <= 0 and phiBar_f(-lambda_min, radius, Lambda, a_j) > 0:
#             sigmaStar = -lambda_min

#             # forms v = (Lambda_one + sigmaStar I)^\dagger P_\parallel^Tg
#             index_pseudo = torch.abs(Lambda + sigmaStar) > tol
#             v = torch.zeros(sizeD + 1)
#             v[index_pseudo] = a_j[index_pseudo] / (Lambda[index_pseudo] + sigmaStar)

#             # forms pStar using Equation (16)
#             if torch.abs(self.gamma + sigmaStar) < tol:
#                 pStar = -P_parallel @ v[:sizeD]
#             else:
#                 pStar = -P_parallel @ v[:sizeD] + (1 / (self.gamma + sigmaStar)) * (self.Psi @ (torch.linalg.solve(PsiPsi, Psig))) - (g / (self.gamma + sigmaStar))

#             if lambda_min < 0:
#                 alpha = torch.sqrt(radius ** 2 - torch.norm(pStar) ** 2)
#                 pHatStar = pStar

#                 # compute z* using Equation (17)
#                 if torch.abs(lambda_min - Lambda[0]) < tol:
#                     zstar = (1 / torch.norm(P_parallel[:, 0])) * alpha * P_parallel[:, 0]
#                 else:
#                     e = torch.zeros(g.size(0))
#                     found = False
#                     for i in range(sizeD):
#                         e[i] = 1
#                         u_min = e - P_parallel @ P_parallel[i]
#                         if torch.norm(u_min) > tol:
#                             found = True
#                             break
#                         e[i] = 0
#                     if not found:
#                         e[sizeD] = 1
#                         u_min = e - P_parallel @ P_parallel[sizeD]
#                     u_min = u_min / torch.norm(u_min)
#                     zstar = alpha * u_min

#                 pStar = pHatStar + zstar
#         else:
#             if lambda_min > 0:
#                 sigmaStar = Newton(0.0, maxIter, tol, radius, Lambda, a_j)
#             else:
#                 sigmaHat = torch.max(a_j / radius - Lambda)
#                 if sigmaHat > -lambda_min:
#                     sigmaStar = Newton(sigmaHat, maxIter, tol, radius, Lambda, a_j)
#                 else:
#                     sigmaStar = Newton(-lambda_min, maxIter, tol, radius, Lambda, a_j)
#             pStar = ComputeSBySMW(self.gamma + sigmaStar, g, Psig, self.Psi, invM, PsiPsi)

#         # Optimality check
#         if show >= 1:
#             BpStar = self.gamma * pStar + self.Psi @ (torch.linalg.solve(invM, Psig))
#             opt1 = torch.norm(BpStar + sigmaStar * pStar + g)
#             opt2 = sigmaStar * torch.abs(radius - torch.norm(pStar))
#             spd_check = lambda_min + sigmaStar
#             if show == 2:
#                 print(f'Optimality condition #1: {opt1:.3e}')
#                 print(f'Optimality condition #2: {opt2:.3e}')
#                 print(f'lambda_min+sigma*: {spd_check:.2e}')
#                 print('\n')
#         else:
#             opt1 = None
#             opt2 = None
#             spd_check = None
#             phiBar_check = None

#         return sigmaStar, pStar, opt1, opt2, spd_check


#     def ComputeSBySMW(tauStar, g, Psig, self.Psi, invM, PsiPsi):
#         vw = tauStar ** 2 * invM + tauStar * PsiPsi
#         pStar = -g / tauStar + self.Psi @ torch.linalg.solve(vw, Psig)
#         return pStar


#     def phiBar_f(sigma, radius, D, a_j):
#         m = a_j.size(0)
#         D = D + sigma * torch.ones(m)
#         eps_tol = 1e-10

#         if (torch.sum(torch.abs(a_j) < eps_tol) > 0) or (torch.sum(torch.abs(D) < eps_tol) > 0):
#             pnorm2 = 0
#             for i in range(m):
#                 if (torch.abs(a_j[i]) > eps_tol) and (torch.abs(D[i]) < eps_tol):
#                     return -1 / radius
#                 elif (torch.abs(a_j[i]) > eps_tol) and (torch.abs(D[i]) > eps_tol):
#                     pnorm2 += (a_j[i] / D[i]) ** 2
#             phiBar = torch.sqrt(1 / pnorm2) - 1 / radius
#             return phiBar

#         p = a_j / D
#         normP = torch.norm(p)
#         phiBar = 1 / normP - 1 / radius
#         return phiBar

#     def phiBar_fg(sigma, radius, D, a_j):
#         m = a_j.size(0)
#         D = D + sigma * torch.ones(m)
#         eps_tol = 1e-10
#         phiBar_g = 0

#         if (torch.sum(torch.abs(a_j) < eps_tol) > 0) or (torch.sum(torch.abs(D) < eps_tol) > 0):
#             pnorm2 = 0
#             for i in range(m):
#                 if (torch.abs(a_j[i]) > eps_tol) and (torch.abs(D[i]) < eps_tol):
#                     phiBar = -1 / radius
#                     phiBar_g = 1 / eps_tol
#                     return phiBar, phiBar_g
#                 elif torch.abs(a_j[i]) > eps_tol and torch.abs(D[i]) > eps_tol:
#                     pnorm2 += (a_j[i] / D[i]) ** 2
#                     phiBar_g += (a_j[i] ** 2) / (D[i] ** 3)

#             normP = torch.sqrt(pnorm2)
#             phiBar = 1 / normP - 1 / radius
#             phiBar_g = phiBar_g / (normP ** 3)
#             return phiBar, phiBar_g

#         # Numerators and denominators are all nonzero
#         p = a_j / D
#         normP = torch.norm(p)
#         phiBar = 1 / normP - 1 / radius

#         phiBar_g = torch.sum((a_j ** 2) / (D ** 3))
#         phiBar_g = phiBar_g / (normP ** 3)

#         return phiBar, phiBar_g


#     def Newton(x0, maxIter, tol, radius, Lambda, a_j):
#         x = x0
#         k = 0

#         f, g = phiBar_fg(x, radius, Lambda, a_j)
#         while (torch.abs(f) > tol) and (k < maxIter):
#             x = x - f / g
#             f, g = phiBar_fg(x, radius, Lambda, a_j)
#             k += 1

#         return x


#     # Example usage:
#     # Define g, S, Y, radius, self.gamma
#     # sigmaStar, pStar, opt1, opt2, spd_check = obs(g, S, Y, radius, self.gamma)
#     # You need to replace g, S, Y, radius, self.gamma with appropriate tensors.
