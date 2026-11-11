# Vendored into SynthScope from NSG-VD (github.com/ZSHsh98/NSG-VD @ 9a60f57; Apache-2.0, Copyright the NSG-VD authors).
#   * ``Pdist2`` and ``MMD_batch2``: utils/mmd_utils.py lines 230-240 and 252-295, unchanged (tabs kept). MMD_batch2 is the
#     official test statistic (test_dMMD.py): one row per test video, the one-sample MMD between that video's features
#     and the fixed reference set, P(fake) decided by ``> 1``.
#   * ``deep_MMD``: models/deep_mmd.py, the parameter container the released checkpoints were saved from: its
#     ``__init__`` and ``set_info`` unchanged. Its training ``forward`` (MMDu + h1_mean_var_gram, train_dMMD.py) is left
#     out, with the I3DDiscriminator import: SynthScope does not train NSG-VD.
import torch
import torch.nn as nn


def Pdist2(x, y):
	"""compute the paired distance between x and y."""
	x_norm = (x ** 2).sum(1).view(-1, 1)
	if y is not None:
		y_norm = (y ** 2).sum(1).view(1, -1)
	else:
		y = x
		y_norm = x_norm.view(1, -1)
	Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
	Pdist[Pdist<0]=0
	return Pdist

def MMD_batch2(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True, coeff_xy=2):
	X = Fea[0:len_s, :] # 400,300
	Y = Fea[len_s:, :] # 2000,300
	if is_smooth:
		X_org = Fea_org[0:len_s, :] # 400,76800
		Y_org = Fea_org[len_s:, :]  # 2000,76800
	L = 1 # generalized Gaussian (if L>1)

	nx = X.shape[0] # 400
	ny = Y.shape[0] # 2000
	Dxx = Pdist2(X, X) # 400,400
	Dyy = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device) # 2000,1  0
	# Dyy = Pdist2(Y, Y) # 2000, 1  0
	Dxy = Pdist2(X, Y).transpose(0,1) # 400，2000 => 2000, 400
	if is_smooth:
		Dxx_org = Pdist2(X_org, X_org)
		Dyy_org = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device) # 2000,1  0
		# Dyy_org = Pdist2(Y_org, Y_org) # 1，1  0
		Dxy_org = Pdist2(X_org, Y_org).transpose(0,1)  # 400，2000 => 2000, 400
	# K_Ix = torch.eye(nx).cuda() # 创建单位矩阵
	# K_Iy = torch.eye(ny).cuda() 

	if is_smooth:
		Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma) # 400,400
		Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma) # 2000,1     值 1
		Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)# 400,2000
	else:
		Kx = torch.exp(-Dxx / sigma0)
		Ky = torch.exp(-Dyy / sigma0)
		Kxy = torch.exp(-Dxy / sigma0)

	nx = Kx.shape[0] # 400
	# ny = 1 # 2000
	is_unbiased = False
	if 1:
		xx = torch.div((torch.sum(Kx)), (nx * nx)) # 400,400 => 1
		yy = Ky.reshape(-1) # 2000
		# yy = torch.div((torch.sum(Ky)), (ny * ny)) # 2000,1 => 2000 1  [1]
		# one-sample U-statistic.

		xy = torch.div(torch.sum(Kxy, dim = 1), (nx )) # 2000
		
		mmd2 = xx - 2 * xy + yy
	return mmd2


class deep_MMD(nn.Module):
    def __init__(self, discriminator, sigma=1000, sigma0=0.1, epsilon=10, img_size=224, is_yy_zero=False, is_smooth=True):
        super(deep_MMD, self).__init__()
        self.img_size = img_size
        self.is_yy_zero = is_yy_zero
        self.is_smooth = is_smooth
        self.epsilonOPT = nn.Parameter(torch.log(torch.rand(1) * 10 ** (-epsilon)), requires_grad=True)
        self.sigmaOPT = nn.Parameter(torch.sqrt(torch.tensor(2 * self.img_size * self.img_size * sigma)), requires_grad=True)
        self.sigma0OPT = nn.Parameter(torch.sqrt(torch.tensor(sigma0)), requires_grad=True)
        
        self.register_buffer('mmd_value', torch.tensor(0.0)) 
        self.register_buffer('ep', torch.tensor(0.0))
        self.register_buffer('sigma', torch.tensor(0.0))
        self.register_buffer('sigma0_u', torch.tensor(0.0))
        self.register_buffer('STAT_u', torch.tensor(0.0))
        
        self.net = discriminator

    def set_info(self, mmd_value, ep, sigma, sigma0_u, STAT_u):
        self.mmd_value.copy_(torch.tensor(mmd_value))
        self.ep.copy_(torch.tensor(ep))
        self.sigma.copy_(torch.tensor(sigma))
        self.sigma0_u.copy_(torch.tensor(sigma0_u))
        self.STAT_u.copy_(torch.tensor(STAT_u))
