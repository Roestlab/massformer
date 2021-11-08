import torch as th
import torch.nn.functional as F
import math
import numpy as np

EPS = np.finfo(np.float32).eps
kl = lambda p,q: th.sum(p*(th.log(p+EPS)-th.log(q+EPS)),dim=1)
mse = lambda x,y: 0.5*th.mean((x-y)**2,dim=1)
normal_nll = lambda x,y: -th.sum(-math.log(0.08*math.sqrt(2*math.pi))-0.5*(x-y)**2/(0.08**2), dim=1)
ln2 = math.log(2)

def compute_weights(x):

	mass_pow = 0.5 # 1.0
	weights = th.arange(1,x.shape[1]+1,device=x.device,dtype=th.float32)**mass_pow
	weights = (weights/th.sum(weights)).unsqueeze(0)
	return weights

def get_loss_func(loss_type):
	
	# set up loss function
	if loss_type == "mse":
		loss_func = mse
	elif loss_type == "w_mse":
		def w_mse(pred,targ):
			weights = compute_weights(pred)		
			return th.sum(weights*(pred-targ)**2,dim=1)
		loss_func = w_mse
	elif loss_type == "js":
		def js(pred,targ):
			pred = F.normalize(pred,dim=1,p=1)
			targ = F.normalize(targ,dim=1,p=1)
			z = 0.5*(pred+targ)
			# relu is to prevent NaN from small negative values
			return th.sqrt(F.relu(0.5*kl(pred,z)+0.5*kl(targ,z)))
		loss_func = js
	elif loss_type == "forw_kl":
		# p=targ, q=pred
		def forw_kl(pred,targ):
			pred = F.normalize(pred,dim=1,p=1)
			targ = F.normalize(targ,dim=1,p=1)
			return kl(targ,pred)
		loss_func = forw_kl
	elif loss_type == "rev_kl":
		# p=pred, q=targ
		def rev_kl(pred,targ):
			pred = F.normalize(pred,dim=1,p=1)
			targ = F.normalize(targ,dim=1,p=1)
			return kl(pred,targ)
		loss_func = rev_kl
	elif loss_type == "normal_nll":
		# TBD change this back
		def myloss(input,target):
			from torch.distributions import Normal
			# normalized_input = F.normalize(input, p = 2, dim = 1)
			# normalized_target = F.normalize(target, p = 2, dim = 1)
			nd = Normal(input,0.08)
			pdf_term = nd.log_prob(target)
			return -th.sum(pdf_term,dim=1)
		loss_func = myloss
	elif loss_type == "wass":
		def wass(pred,targ):
			# does not care about actual m/z distances (just a constant multipler)
			pred = F.normalize(pred,dim=1,p=1)
			targ = F.normalize(targ,dim=1,p=1)
			pred_cdf = th.cumsum(pred,dim=1)
			targ_cdf = th.cumsum(targ,dim=1)
			return th.sum(th.abs(pred_cdf-targ_cdf),dim=1)
		loss_func = wass
	elif loss_type == "cos":
		def cos(pred,targ):
			pred = F.normalize(pred,dim=1,p=2).unsqueeze(1)
			targ = F.normalize(targ,dim=1,p=2).unsqueeze(2)
			return 1.-th.matmul(pred,targ).squeeze(-1).squeeze(-1)
		loss_func = cos
	elif loss_type == "w_cos":
		def w_cos(pred,targ):
			weights = compute_weights(pred)
			w_pred = F.normalize(weights*pred,dim=1,p=2).unsqueeze(1)
			w_targ = F.normalize(weights*targ,dim=1,p=2).unsqueeze(2)
			return 1.-th.matmul(w_pred,w_targ).squeeze(-1).squeeze(-1)
		loss_func = w_cos
	else:
		raise NotImplementedError

	return loss_func

def get_sim_func(sim_type):

	if sim_type == "cos":
		def cos(pred,targ):
			n_pred = F.normalize(pred,p=2,dim=1).unsqueeze(1)
			n_targ = F.normalize(targ,p=2,dim=1).unsqueeze(2)
			return th.bmm(n_pred,n_targ).squeeze(-1).squeeze(-1)
		sim_func = cos
	elif sim_type == "w_cos":
		def w_cos(pred,targ):
			weights = compute_weights(pred)
			n_pred = F.normalize(weights*pred,p=2,dim=1).unsqueeze(1)
			n_targ = F.normalize(weights*targ,p=2,dim=1).unsqueeze(2)
			return th.bmm(n_pred,n_targ).squeeze(-1).squeeze(-1)
		sim_func = w_cos
	elif sim_type == "wass":
		def wass(pred,targ):
			pred = F.normalize(pred,dim=1,p=1)
			targ = F.normalize(targ,dim=1,p=1)
			pred_cdf = th.cumsum(pred,dim=1)
			targ_cdf = th.cumsum(targ,dim=1)
			max_val = float(pred.shape[1]-1)
			return (max_val-th.sum(th.abs(pred_cdf-targ_cdf),dim=1))/max_val
		sim_func = wass
	elif sim_type == "s_dot":
		def s_dot(pred,targ):
			pred = (pred**0.5).unsqueeze(1)
			targ = (targ**0.5).unsqueeze(2)
			numerator = th.matmul(pred,targ).squeeze(-1).squeeze(-1)
			denominator = th.sum(pred,dim=1) * th.sum(targ,dim=1)
			return numerator / denominator
		sim_func = s_dot
	elif sim_type == "w_s_dot":
		def w_s_dot(pred,targ):
			weights = compute_weights(pred)
			pred = ((weights*pred)**0.5).unsqueeze(1)
			targ = ((weights*targ)**0.5).unsqueeze(2)
			numerator = th.matmul(pred,targ).squeeze(-1).squeeze(-1)
			denominator = th.sum(pred,dim=1) * th.sum(targ,dim=1)
			return numerator / denominator
		sim_func = w_s_dot
	elif sim_type == "js":
		def js(pred,targ):
			pred = F.normalize(pred,dim=1,p=1)
			targ = F.normalize(targ,dim=1,p=1)
			z = 0.5*(pred+targ)
			return ln2-(0.5*kl(pred,z)+0.5*kl(targ,z))
		sim_func = js
	else:
		raise ValueError

	return sim_func

