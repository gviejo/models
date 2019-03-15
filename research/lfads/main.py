# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lfads import LFADS
import numpy as np
import os
import tensorflow as tf
import re
import utils
import sys
import h5py
from functions import *
from time import time

hps = hps_dict_to_obj({
	"data_dir":                     'data/',        #"Data for training"    
	"data_filename_stem":           'this_data',            #"Filename stem for data dictionaries."    
	"lfads_save_dir":               'data/lfads_chaotic_rnn_no_inputs', #"model save dir"    
	"kind":                         "train",      #"Type of model to build {train, posterior_sample_and_average, posterior_push_mean, prior_sample, write_model_params"    
	"output_dist":                  'poisson',    #"Type of output distribution, 'poisson' or 'gaussian'"  
	"allow_gpu_growth":             False,        #"If true, only allocate amount of memory needed for Session. Otherwise, use full GPU memory."    
	"checkpoint_pb_load_name":      'checkpoint', #"Name of checkpoint files, use 'checkpoint_lve' for best error" 
	"checkpoint_name":              'lfads_vae',  #"Name of checkpoint files (.ckpt appended)" 
	"output_filename_stem":         '',           #"Name of output file (postfix will be added)"   
	"device":                       'gpu:0',      #"Which device to use (default: \"gpu:0\", can also be \"cpu:0\", \"gpu:1\", etc)"   
	"csv_log":                      'fitlog',     #"Name of file to keep running log of fit likelihoods, etc (.csv appended)"  
	"max_ckpt_to_keep":             5,            #"Max # of checkpoints to keep (rolling)" 
	"ps_nexamples_to_process":      sys.maxsize,  #"Number of examples to process for posterior sample and average (not number of samples to average over)."    
	"max_ckpt_to_keep_lve":         5,            #"Max # of checkpoints to keep for lowest validation error models (rolling)"  
	"ext_input_dim":                0,            #"Dimension of external inputs"   
	"num_steps_for_gen_ic":         sys.maxsize,  #"Number of steps to train the generator initial conditon."   
	"inject_ext_input_to_gen":      False,        #"Should observed inputs be input to model via encoders, or injected directly into generator?"    
	"cell_weight_scale":            1.0,          #"Input scaling for input weights in generator."    
	"ic_dim":                       64,           #"Dimension of h0"    
	"factors_dim":                  50,           #"Number of factors from generator"   
	"ic_enc_dim":                   128,          #"Cell hidden size, encoder of h0"    
	"gen_dim":                      200,          #"Cell hidden size, generator."   
	"gen_cell_input_weight_scale":  1.0,          #"Input scaling for input weights in generator."    
	"gen_cell_rec_weight_scale":    1.0,          #"Input scaling for rec weights in generator."  
	"ic_prior_var_min":             0.1,          #"Minimum variance in posterior h0 codes."  
	"ic_prior_var_scale":           0.1,          #"Variance of ic prior distribution"    
	"ic_prior_var_max":             0.1,          #"Maximum variance of IC prior distribution."   
	"ic_post_var_min":              0.0001,       #"Minimum variance of IC posterior distribution."   
	"co_prior_var_scale":           0.1,          #"Variance of control input prior distribution."    
	"prior_ar_atau":                10.0,         #"Initial autocorrelation of AR(1) priors." 
	"prior_ar_nvar":                0.1,          #"Initial noise variance for AR(1) priors." 
	"do_train_prior_ar_atau":       True,         #"Is the value for atau an init: or the constant value?"  
	"do_train_prior_ar_nvar":       True,         #"Is the value for noise variance an init, or the constant value?"    
	"co_dim":                       1,            #"Number of control net outputs (>0 builds that graph)."  
	"do_causal_controller":         False,        #"Restrict the controller create only causal inferred inputs?"    
	"do_feed_factors_to_controller": True,        #"Should factors[t-1] be input to controller at time t?"  
	"feedback_factors_or_rates":    'factors',    #"Feedback the factors or the rates to the controller? Acceptable values: 'factors' or 'rates'." 
	"controller_input_lag":         1,            #"Time lag on the encoding to controller t-lag for forward, t+lag for reverse."   
	"ci_enc_dim":                   128,          #"Cell hidden size: encoder of control inputs"    
	"con_dim":                      128,          #"Cell hidden size, controller"   
	"batch_size":                   5,            #"Batch size to use during training." 
	"learning_rate_init":           0.01,         #"Learning rate initial value"  
	"learning_rate_decay_factor":   0.95,         #"Learning rate decay, decay by this fraction every so often."  
	"learning_rate_stop":           0.005,        #"The lr is adaptively reduced, stop training at this value."   
	"learning_rate_n_to_compare":   2,            #"Number of previous costs current cost has to be worse than, to lower learning rate."    
	"max_grad_norm":                200.0,        #"Max norm of gradient before clipping."    
	"cell_clip_value":              5.0,          #"Max value recurrent cell can take before being clipped."  
	"do_train_io_only":             False,        #"Train only the input (readin) and output (readout) affine functions."   
	"do_train_encoder_only":        False,        #"Train only the encoder weights."    
	"do_reset_learning_rate":       False,        #"Reset the learning rate to initial value."  
	"do_train_readin":              True,         #"Whether to train the readin matrices and bias vectors. False leaves them fixed at their initial values specified by the alignment matrices and vectors."    
	"keep_prob":                    0.95,         #"Dropout keep probability."    
	"temporal_spike_jitter_width":  0,            #"Shuffle spikes around this window." 
	"l2_gen_scale":                 2000.0,       #"L2 regularization cost for the generator only."   
	"l2_con_scale":                 0.0,          #"L2 regularization cost for the controller only."  
	"co_mean_corr_scale":           0.0,          #"Cost of correlation (thru time)in the means of controller output."    
	"kl_ic_weight":                 1.0,          #"Strength of KL weight on initial conditions KL penatly."  
	"kl_co_weight":                 1.0,          #"Strength of KL weight on controller output KL penalty."   
	"kl_start_step":                0,            #"Start increasing weight after this many steps." 
	"kl_increase_steps":            2000,         #"Increase weight of kl cost to avoid local minimum." 
	"l2_start_step":                0,            #"Start increasing l2 weight after this many steps."  
	"l2_increase_steps":            2000,         #"Increase weight of l2 cost to avoid local minimum." 
})

t1 = time()
#####################################################################################
# load_datasets
#####################################################################################
fnames = os.listdir(hps.data_dir)

datasets = {}
  
for fname in fnames:
	if fname.startswith(hps.data_filename_stem):
		with h5py.File(os.path.join(hps.data_dir,fname), 'r') as hf:
			data_dict = {k: np.array(v) for k, v in hf.items()}
		
		idx = len(hps.data_filename_stem) + 1
		key = fname[idx:]
		data_dict['data_dim'] = data_dict['train_data'].shape[2]
		data_dict['num_steps'] = data_dict['train_data'].shape[1]        
		# clean data_dict
		for k in ['train_truth', 'train_ext_input', 'valid_data','valid_truth', 'valid_ext_input', 'valid_train']:
			if k not in data_dict:
				data_dict[k] = None                
		datasets[key] = data_dict


hps.dataset_names = list(datasets.keys())
hps.dataset_dims = {k:datasets[k]['data_dim'] for k in datasets}
hps.num_steps = datasets[list(datasets.keys())[0]]['num_steps']
hps.ndatasets = len(hps.dataset_names)
has_any_valid_set = True


if hps.num_steps_for_gen_ic > hps.num_steps: hps.num_steps_for_gen_ic = hps.num_steps

#####################################################################################
# train
#####################################################################################
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as session:

	#####################################################################################
	# build_model(hps, kind='train', datasets = datasets)
	#####################################################################################
	with tf.variable_scope("LFADS", reuse=None):
		model = LFADS(hps, kind='train', datasets=datasets) 
		
	tf.global_variables_initializer().run()
	session.run(model.learning_rate.initializer)
	
	#####################################################################################
	# model.train_model(datasets)
	#####################################################################################
	
	lr = session.run(model.learning_rate)
	lr_stop = hps.learning_rate_stop
	
	train_costs = []
	valid_costs = []
	learning_rates = []        
	
	while True:		
		learning_rates.append(lr)
		#####################################################################################          
		# self.train_epochs(datasets, do_save_ckpt=do_save_ckpt)
		#####################################################################################
		ops_to_eval = [model.cost, model.recon_cost, model.kl_cost, model.kl_weight, model.l2_cost, model.l2_weight, model.train_op]

		#####################################################################################
		# self.run_epochs(datasets, ops_to_eval, kind="train")
		#####################################################################################
		all_name_example_idx_pairs = model.shuffle_and_flatten_datasets(datasets, hps.kind)
		collected_op_values = np.zeros((6,len(all_name_example_idx_pairs)))
		for j, (name, example_idxs) in enumerate(all_name_example_idx_pairs):
			data_dict = datasets[name]
			data_bxtxd, ext_input_bxtxi = model.get_batch(data_dict['train_data'], data_dict['train_ext_input'],example_idxs=example_idxs)
			feed_dict = model.build_feed_dict(name, data_bxtxd, ext_input_bxtxi, keep_prob=None)
			evaled_ops_np = session.run(ops_to_eval, feed_dict=feed_dict)
			collected_op_values[:,j] = np.array(evaled_ops_np[0:6])
		#####################################################################################
		
		mean_cost = collected_op_values.mean(1)
		tr_total_cost = mean_cost[0]
		#####################################################################################

		#####################################################################################
		# self.eval_cost_epoch(datasets, kind='valid')
		#####################################################################################
		ops_to_eval = [model.cost, model.recon_cost, model.kl_cost]

		#####################################################################################
		# self.run_epochs(datasets, ops_to_eval, kind="valid", keep_prob = 1.0)
		#####################################################################################
		all_name_example_idx_pairs = model.shuffle_and_flatten_datasets(datasets, 'valid')
		collected_op_values = np.zeros((3,len(all_name_example_idx_pairs)))
		for j, (name, example_idxs) in enumerate(all_name_example_idx_pairs):
			data_dict = datasets[name]
			data_bxtxd, ext_input_bxtxi = model.get_batch(data_dict['valid_data'], data_dict['valid_ext_input'],example_idxs=example_idxs)
			feed_dict = model.build_feed_dict(name, data_bxtxd, ext_input_bxtxi, keep_prob=1.0)
			evaled_ops_np = session.run(ops_to_eval, feed_dict=feed_dict)            
			collected_op_values[:,j] = np.array(evaled_ops_np[0:3])
		#####################################################################################
		
		mean_cost = collected_op_values.mean(1)
		ev_total_cost = mean_cost[0]
		#####################################################################################

		valid_costs.append(ev_total_cost)

		# Manage learning rate.        
		n_lr = hps.learning_rate_n_to_compare        
		if len(train_costs) > n_lr and tr_total_cost > np.max(train_costs[-n_lr:]):            
			lr = session.run(model.learning_rate_decay_op)            
			print("     Decreasing learning rate to %f." % lr)
			# Force the system to run n_lr times while at this lr.
			train_costs.append(np.inf)
		else:
			train_costs.append(tr_total_cost)

		if lr < lr_stop:
			print("Stopping optimization based on learning rate criteria.")
			break
	#####################################################################################

print(time()-t1)


sys.exit()
#######################################################################################
# POSTERIOR SAMPLE AND AVERAGE
#
# 	write_model_runs(write_model_runs(hps, datasets, hps.output_filename_stem, push_mean=False))
#			model.write_model_runs(datasets, output_fname, push_mean)
#######################################################################################
model.hps.kind = 'posterior_sample_and_average'

for data_name, data_dict in datasets.items():
	data_tuple = [('train', data_dict['train_data'], data_dict['train_ext_input']), ('valid', data_dict['valid_data'], data_dict['valid_ext_input'])]
	for data_kind, data_extxd, ext_input_extxi in data_tuple:
		fname = "model_runs_" + data_name + '_' + data_kind + '_' + model.hps.kind

		###############################################################################
		# model.eval_model_runs_avg_epoch
		###############################################################################		
	    hps = self.hps
	    batch_size = hps.batch_size
	    E, T, D  = data_extxd.shape
	    E_to_process = hps.ps_nexamples_to_process
	    if E_to_process > E:
	      E_to_process = E

	    if hps.ic_dim > 0:
	      prior_g0_mean = np.zeros([E_to_process, hps.ic_dim])
	      prior_g0_logvar = np.zeros([E_to_process, hps.ic_dim])
	      post_g0_mean = np.zeros([E_to_process, hps.ic_dim])
	      post_g0_logvar = np.zeros([E_to_process, hps.ic_dim])

	    if hps.co_dim > 0:
	      controller_outputs = np.zeros([E_to_process, T, hps.co_dim])
	    gen_ics = np.zeros([E_to_process, hps.gen_dim])
	    gen_states = np.zeros([E_to_process, T, hps.gen_dim])
	    factors = np.zeros([E_to_process, T, hps.factors_dim])

	    if hps.output_dist == 'poisson':
	      out_dist_params = np.zeros([E_to_process, T, D])
	    elif hps.output_dist == 'gaussian':
	      out_dist_params = np.zeros([E_to_process, T, D+D])
	    else:
	      assert False, "NIY"

	    costs = np.zeros(E_to_process)
	    nll_bound_vaes = np.zeros(E_to_process)
	    nll_bound_iwaes = np.zeros(E_to_process)
	    train_steps = np.zeros(E_to_process)
	    for es_idx in range(E_to_process):
	      print("Running %d of %d." % (es_idx+1, E_to_process))
	      example_idxs = es_idx * np.ones(batch_size, dtype=np.int32)
	      data_bxtxd, ext_input_bxtxi = self.get_batch(data_extxd,
	                                                   ext_input_extxi,
	                                                   batch_size=batch_size,
	                                                   example_idxs=example_idxs)
	      model_values = self.eval_model_runs_batch(data_name, data_bxtxd,
	                                                ext_input_bxtxi,
	                                                do_eval_cost=True,
	                                                do_average_batch=True)

	      if self.hps.ic_dim > 0:
	        prior_g0_mean[es_idx,:] = model_values['prior_g0_mean']
	        prior_g0_logvar[es_idx,:] = model_values['prior_g0_logvar']
	        post_g0_mean[es_idx,:] = model_values['post_g0_mean']
	        post_g0_logvar[es_idx,:] = model_values['post_g0_logvar']
	      gen_ics[es_idx,:] = model_values['gen_ics']

	      if self.hps.co_dim > 0:
	        controller_outputs[es_idx,:,:] = model_values['controller_outputs']
	      gen_states[es_idx,:,:] = model_values['gen_states']
	      factors[es_idx,:,:] = model_values['factors']
	      out_dist_params[es_idx,:,:] = model_values['output_dist_params']
	      costs[es_idx] = model_values['costs']
	      nll_bound_vaes[es_idx] = model_values['nll_bound_vaes']
	      nll_bound_iwaes[es_idx] = model_values['nll_bound_iwaes']
	      train_steps[es_idx] = model_values['train_steps']
	      print('bound nll(vae): %.3f, bound nll(iwae): %.3f' \
	            % (nll_bound_vaes[es_idx], nll_bound_iwaes[es_idx]))

	    model_runs = {}
	    if self.hps.ic_dim > 0:
	      model_runs['prior_g0_mean'] = prior_g0_mean
	      model_runs['prior_g0_logvar'] = prior_g0_logvar
	      model_runs['post_g0_mean'] = post_g0_mean
	      model_runs['post_g0_logvar'] = post_g0_logvar
	    model_runs['gen_ics'] = gen_ics

	    if self.hps.co_dim > 0:
	      model_runs['controller_outputs'] = controller_outputs
	    model_runs['gen_states'] = gen_states
	    model_runs['factors'] = factors
	    model_runs['output_dist_params'] = out_dist_params
	    model_runs['costs'] = costs
	    model_runs['nll_bound_vaes'] = nll_bound_vaes
	    model_runs['nll_bound_iwaes'] = nll_bound_iwaes
	    model_runs['train_steps'] = train_steps
	    return model_runs
			

		###############################################################################
		full_fname = os.path.join(hps.lfads_save_dir, fname)
		write_data(full_fname, model_runs, compression='gzip')
		print("Done.")


# model.write_model_runs(datasets)

# def write_model_runs(hps, datasets, output_fname=None, push_mean=False):
# 	"""Run the model on the data in data_dict, and save the computed values.

# 	LFADS generates a number of outputs for each examples, and these are all
# 	saved.  They are:
# 	The mean and variance of the prior of g0.
# 	The mean and variance of approximate posterior of g0.
# 	The control inputs (if enabled)
# 	The initial conditions, g0, for all examples.
# 	The generator states for all time.
# 	The factors for all time.
# 	The rates for all time.

# 	Args:
# 	hps: The dictionary of hyperparameters.
# 	datasets: A dictionary of data dictionaries.  The dataset dict is simply a
# 	  name(string)-> data dictionary mapping (See top of lfads.py).
# 	output_fname (optional): output filename stem to write the model runs.
# 	push_mean: if False (default), generates batch_size samples for each trial
# 	  and averages the results. if True, runs each trial once without noise,
# 	  pushing the posterior mean initial conditions and control inputs through
# 	  the trained model. False is used for posterior_sample_and_average, True
# 	  is used for posterior_push_mean.
# 	"""
# 	model = build_model(hps, kind=hps.kind, datasets=datasets)
# 	model.write_model_runs(datasets, output_fname, push_mean)
# 	return




# POSTERIOR PUSH MEAN

# PRIOR SAMPLE

# WRITE MODEL PARAMS


# with sess.as_default():
#     with tf.device(hps.device):
#         if kind == "train":
#             train(hps, datasets)
#         elif kind == "posterior_sample_and_average":
#             write_model_runs(hps, datasets, hps.output_filename_stem, push_mean=False)
#         elif kind == "posterior_push_mean":
#             write_model_runs(hps, datasets, hps.output_filename_stem, push_mean=True)
#         elif kind == "prior_sample":
#             write_model_samples(hps, datasets, hps.output_filename_stem)
#         elif kind == "write_model_params":
#             write_model_parameters(hps, hps.output_filename_stem, datasets)
#         else:
#             assert False, ("Kind %s is not implemented. " % kind)
