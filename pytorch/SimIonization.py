import torch

class SimIonization(torch.nn.Module):
    def __init__(self, test_case=0, std_in=0.1):
        super(SimIonization, self).__init__()
        
        # CL: make life_time the same scale as the other variables so it can be more easily learned
        self.scale_lifetime = 10000.0

        # Known parameters for genearting ground truth output
        self._density = 1.38    # g/cm^3
        self._alpha   = 0.847
        self._beta    = 0.2061
        self._efield  = 0.500   # V/sm
        self._lifetime = 6000 / self.scale_lifetime   # ms
        self._energy_threshold = 0.06 # MeV threshold to ignore drift
        self._dedx_threshold   = 0.0001 # MeV/cm threshold to ignore ...
        self._vdrift  = 0.153812 # cm/us

        # Variables for recovering the above model parameters
        self._v_density = torch.nn.Parameter(torch.randn(()))
        
        if test_case == 0:
            self._v_alpha = torch.tensor(0.847, requires_grad=False)
            self._v_beta = torch.tensor(0.2061, requires_grad=False)
        elif test_case == 1:
            self._v_beta = torch.nn.Parameter(torch.rand(()))
            self._v_alpha = torch.nn.Parameter(torch.rand(()))
        else:            
            self._v_alpha = torch.nn.Parameter(torch.normal(mean=0.847, std=std_in, size=()))
            self._v_beta = torch.nn.Parameter(torch.normal(mean=0.2061, std=std_in, size=()))            

        self._v_efield = torch.nn.Parameter(torch.rand(()))
        self._v_lifetime = torch.nn.Parameter(torch.normal(mean=0.6, std=std_in, size=()))   
        self._v_energy_threshold = torch.nn.Parameter(torch.rand(()))
        self._v_dedx_threshold = torch.nn.Parameter(torch.rand(()))
        self._v_vdrift = torch.nn.Parameter(torch.rand(()))

        self._i_alpha = torch.clone(self._v_alpha).detach()
        self._i_beta = torch.clone(self._v_beta).detach()
        self._i_vdrift = torch.clone(self._v_vdrift).detach()
        self._i_lifetime = torch.clone(self._v_lifetime).detach()


    def combine(self, x, alpha, beta, vdrift, lifetime):
        # apply recombination
        x_0 = x[:,0] * torch.log(alpha + beta * x[:,2]) / (beta * x[:,2])
        # apply lifetime (with scale)
        x_0 = x_0 * torch.exp( -1. * x[:,1] / vdrift / (lifetime * self.scale_lifetime))
        return torch.cat((x_0.unsqueeze(-1), x[:,1].unsqueeze(-1), x[:,2].unsqueeze(-1)), 1)

    def forward(self, x, task):
        '''
        Input: x tensor w/ shape (N,3) where 3 = (E,x,de/dx)
        Return: (N,1) where 1 = (Q,x,de/dx)
        '''
        if task == 'generate':
            x_out = self.combine(x, self._alpha, self._beta, self._vdrift, self._lifetime)
        elif task == 'learn':
            x_out = self.combine(x, self._v_alpha, self._v_beta, self._v_vdrift, self._v_lifetime)
        else:
            print('Task {} is invalid'.format(task))
            exit(0)

        return x_out


class SimIonizationData(torch.nn.Module):
    def __init__(self, x):
        super(SimIonizationData, self).__init__()

        # CL: make life_time the same scale as the other variables so it can be more easily learned
        self.scale_lifetime = 10000.0
        
        # Known parameters for genearting ground truth output
        self._density = 1.38    # g/cm^3
        self._alpha   = 0.847
        self._beta    = 0.2061
        self._efield  = 0.500   # V/sm
        self._lifetime = 6000 / self.scale_lifetime   # ms
        self._energy_threshold = 0.06 # MeV threshold to ignore drift
        self._dedx_threshold   = 0.0001 # MeV/cm threshold to ignore ...
        self._vdrift  = 0.153812 # cm/us

        # Ground truth x
        self.x_gt = x
        self.x_gt.requires_grad = False
        self.y_gt = self.combine(self.x_gt)

    def combine(self, x):
        x_0 = x[:,0] * torch.log(self._alpha + self._beta * x[:,2]) / (self._beta * x[:,2])
        x_0 = x_0 * torch.exp( -1. * x[:,1] / self._vdrift / (self._lifetime * self.scale_lifetime))
        return torch.cat((x_0.unsqueeze(-1), x[:,1].unsqueeze(-1), x[:,2].unsqueeze(-1)), 1)

    def forward(self, x):
        return self.combine(x)


