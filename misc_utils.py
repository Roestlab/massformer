import contextlib
import numpy as np
import torch as th
from distutils.util import strtobool
import joblib
import tqdm
import os
import errno
import signal
from functools import wraps
from timeit import default_timer as timer
from datetime import timedelta
import sys

def flatten_lol(list_of_list):
	flat_list = []
	for ll  in list_of_list:
		flat_list.extend(ll)
	return flat_list

@contextlib.contextmanager
def np_temp_seed(seed):
	state = np.random.get_state()
	np.random.seed(seed)
	try:
		yield
	finally:
		np.random.set_state(state)

@contextlib.contextmanager
def th_temp_seed(seed):
	state = th.get_rng_state()
	th.manual_seed(seed)
	try:
		yield
	finally:
		th.set_rng_state(state)

def np_scatter_add(input,axis,index,src):
	""" numpy wrapper for scatter_add """

	th_input = th.as_tensor(input,device="cpu")
	th_index = th.as_tensor(index,device="cpu")
	th_src = th.as_tensor(src,device="cpu")
	dim = axis
	th_output = th.scatter_add(th_input,dim,th_index,th_src)
	output = th_output.numpy()
	return output

def np_one_hot(input,num_classes=None):
	""" numpy wrapper for one_hot """

	th_input = th.as_tensor(input,device="cpu")
	th_oh = th.nn.functional.one_hot(th_input,num_classes=num_classes)
	oh = th_oh.numpy()
	return oh

def list_dict_to_dict_array(list_dict):

	dict_keys = list_dict[0].keys()
	dict_list = {k:[] for k in dict_keys}
	for d in list_dict:
		for k,v in d.items():
			dict_list[k].append(v)
	dict_arr = {}
	for k,v in dict_list.items():
		dict_arr[k] = np.stack(v,axis=0)
	return dict_arr

def df_select_with_dict(df,d):

	masks = []
	for k,v in d.items():
		masks.append(df[k] == v)
	all_mask = masks[0]
	for mask in masks:
		all_mask = all_mask & mask
	return df[all_mask]

def params_to_str(params):

	_params = {}
	for k,v in params.items():
		if isinstance(v,list):
			_params[k] = str(sorted(v))
		else:
			assert isinstance(v,str), v
			_params[k] = v
	params_str = str(_params)
	return _params, params_str

def booltype(x):
	return bool(strtobool(x))

# https://stackoverflow.com/a/58936697/6937913
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
	"""Context manager to patch joblib to report into tqdm progress bar given as argument"""
	class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
		def __init__(self, *args, **kwargs):
			super().__init__(*args, **kwargs)

		def __call__(self, *args, **kwargs):
			tqdm_object.update(n=self.batch_size)
			return super().__call__(*args, **kwargs)

	old_batch_callback = joblib.parallel.BatchCompletionCallBack
	joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
	try:
		yield tqdm_object
	finally:
		joblib.parallel.BatchCompletionCallBack = old_batch_callback
		tqdm_object.close()

# https://stackoverflow.com/questions/2281850/timeout-function-if-it-takes-too-long-to-finish
class TimeoutError(Exception):
	pass

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
	def decorator(func):
		def _handle_timeout(signum, frame):
			raise TimeoutError(error_message)

		def wrapper(*args, **kwargs):
			signal.signal(signal.SIGALRM, _handle_timeout)
			signal.alarm(seconds)
			try:
				result = func(*args, **kwargs)
			finally:
				signal.alarm(0)
			return result

		return wraps(func)(wrapper)

	return decorator

def time_function(func,*args,num_reps=100):

	deltas = []
	for i in range(num_reps):
		start = timer()
		func(*args)
		end = timer()
		delta = timedelta(seconds=end-start)
		deltas.append(delta)
	avg_delta = np.mean(deltas)
	print(avg_delta)

def sharpen(x, p):
	return x**p / th.sum(x**p,dim=1,keepdim=True)

# https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python
@contextlib.contextmanager
def suppress_output(stdout=True,stderr=True):
	with open(os.devnull, "w") as devnull:
		old_stdout = sys.stdout
		old_stderr = sys.stderr
		if stdout:
			sys.stdout = devnull
		if stderr:
			sys.stderr = devnull
		try:  
			yield
		finally:
			sys.stdout = old_stdout
			sys.stderr = old_stderr

def list_str2float(str_list):
	return [float(str_item) for str_item in str_list]

def none_or_nan(thing):
	if thing is None:
		return True
	elif isinstance(thing,float) and np.isnan(thing):
		return True
	else:
		return False

class DummyContext:

	def __enter__(self):
		pass

	def __exit__(self,*args):
		pass

class DummyScaler:

	def scale(self,grad):
		return grad

	def step(self,optimizer):
		optimizer.step()
	
	def update(self):
		pass
