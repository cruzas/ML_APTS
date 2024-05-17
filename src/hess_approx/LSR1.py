import torch


class LSR1:
    def __init__(self, ha_memory=10,device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.memory_init = False
        self.Y = None
        self.S = None
        self.M = None
        self.Psi = None
        self.gamma = torch.tensor([1.0],device=device)
        self.device = device

        self.config = dict(
            ha_type="LSR1",
            ha_memory=ha_memory,
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

    def update_memory_inv(self, s, y):
        self.update_memory(y, s)

    def update_memory(self, s, y):
        Bs = self.apply(s)
        y_Bs = y - Bs

        val = torch.abs(torch.dot(s, y_Bs))
        s_norm = torch.norm(s)
        yBs_norm = torch.norm(y_Bs)

        perform_up = True

        if not torch.isfinite(val) or torch.isnan(val):
            perform_up = False
            if self.print_errors:
                print("L_SR1:: Skipping update2 ... ")

        if not torch.isfinite(val) or val < self.r * s_norm * yBs_norm:
            perform_up = False
            if self.print_errors:
                print("L_SR1:: Skipping update2 ... ")

        if perform_up:
            if not self.memory_init:
                self.S = s.view(len(s), 1)
                self.Y = y.view(len(y), 1)
                self.memory_init = True
            elif self.S.shape[1] < self.memory:
                self.S = torch.cat((self.S, s.view(len(s),1)), dim=1)
                self.Y = torch.cat((self.Y, y.view(len(y),1)), dim=1)
            else:
                self.S[:, 0] = s
                self.Y[:, 0] = y
                self.S = torch.roll(self.S, shifts=(-1), dims=(1))
                self.Y = torch.roll(self.Y, shifts=(-1), dims=(1))

            while self.precompute() and self.S.shape[1] > 0:
                self.S = torch.cat((self.S[:, 1:], torch.zeros_like(self.S[:, :1])), dim=1)
                self.Y = torch.cat((self.Y[:, 1:], torch.zeros_like(self.Y[:, :1])), dim=1)

            if self.S.shape[1] == 0:
                self.memory_init = False

            return self.S, self.Y

    def sample_dir_update_memory_inv(self, closure, model, x, g_old):
        size = len(x)
        for m in range(self.memory):
            p_k = self.sampling_radius * torch.randn(size)

            if torch.dot(p_k, g_old) > 0:
                p_k = -1.0 * p_k

            x_new = x + p_k
            _ = closure()
            g_new = torch.cat([p.grad.flatten() for p in model.parameters()])
            y = g_new - g_old
            self.update_memory_inv(p_k, y)

        return self.S, self.Y

    def sample_dir_update_memory(self, closure, model, x, g_old):
        size = len(x)
        for m in range(self.memory):
            p_k = self.sampling_radius * torch.randn(size)

            if torch.dot(p_k, g_old) > 0:
                m -= 1
                break

            x_new = x + p_k
            _ = closure()
            g_new = torch.cat([p.grad.flatten() for p in model.parameters()])
            y = g_new - g_old

            self.update_memory(p_k, y)

        return self.S, self.Y

    def reset_memory(self):
        self.memory_init = False
        self.S = None
        self.Y = None
        self.M_inv = None
        self.M = None
        self.gamma = torch.tensor([1.0],device=self.device)
        self.Psi = None

    def precompute(self, s=None, y=None):
        if self.S is None or self.Y is None:
            return False
        if s is None or y is None:
            s = self.S[:, -1]
            y = self.Y[:, -1]

        SY = torch.matmul(self.S.transpose(0, 1), self.Y)
        D = torch.diag(SY.diag())
        L = torch.tril(SY, diagonal=-1)
        DLL = D + L + L.transpose(0, 1)

        self.gamma = self.init_eig_min(DLL, s, y)

        Sgamma = self.gamma * self.S
        self.M_inv = DLL - torch.matmul(self.S.transpose(0, 1), Sgamma)

        self.M_inv = (self.M_inv + self.M_inv.transpose(0, 1)) / 2.0

        self.Psi = self.Y - Sgamma

        PsiPsi = torch.matmul(self.Psi.transpose(0, 1), self.Psi)

        try:
            self.M = torch.inverse(self.M_inv)
        except torch.linalg.LinAlgError:
            if self.print_errors:
                print("Problem in update M ---- ")
            return True

        try:
            R = torch.linalg.cholesky(PsiPsi, upper=False)
        except torch.linalg.LinAlgError:
            if self.print_errors:
                print("Problem in update for psi ---- ")
            return True

        try:
            helpp = torch.linalg.solve(self.M_inv, R.transpose(0, 1))
        except torch.linalg.LinAlgError:
            if self.print_errors:
                print("Problem in update M_inv ---- ")
            return True

        return False

    def apply(self, v):
        if not self.memory_init:
            result = self.gamma * v
        else:
            a = torch.matmul(self.Psi.transpose(0, 1), v)
            b = torch.matmul(self.M, a)
            result = (self.gamma * v) + torch.matmul(self.Psi, b)
        return result

    def apply_inv(self, v):
        if not self.memory_init:
            result = self.gamma * v
        else:
            a = torch.matmul(self.Psi.transpose(0, 1), v)
            b = torch.matmul(self.M, a)
            result = (self.gamma * v) + torch.matmul(self.Psi, b)
        return result

    def init_eig_min(self, DLL, s, y):
        gamma_result = torch.tensor([1.0],device=self.device)

        if self.eig_type == "one":
            gamma_result = torch.tensor([1.0],device=self.device)

        elif self.eig_type == "eigen_decomp":
            SS = torch.matmul(self.S.transpose(0, 1), self.S)
            eig_min = torch.tensor([1.0],device=self.device)

            try:
                eigvals = torch.linalg.eigvals(DLL, SS)
                eig_min = torch.min(eigvals)
            except torch.linalg.LinAlgError:
                eig_min = torch.tensor([1.0],device=self.device)

            if eig_min <= 0:
                eig_min = torch.tensor([1.0],device=self.device)
            else:
                eig_min = torch.tensor([0.9],device=self.device) * eig_min

            if torch.abs(eig_min) < self.tol:
                eig_min = torch.tensor([1.0],device=self.device)

            gamma_result = eig_min

        elif self.eig_type == "standard":
            gamma = torch.dot(s, y)
            gamma = gamma / torch.dot(y, y)

            if torch.abs(gamma) < self.tol:
                gamma = torch.tensor([1.0],device=self.device)

            gamma_result = torch.tensor([1.0],device=self.device) / gamma

        else:
            print("------------- not defined type of init_eig_min ---------")
            exit(0)

        if not torch.isfinite(gamma_result) or torch.isnan(gamma_result):
            gamma_result = torch.tensor([1.0],device=self.device)

        return gamma_result
