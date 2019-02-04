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


def build_model(hps, kind="train", datasets=None):
  """Builds a model from either random initialization, or saved parameters.

  Args:
    hps: The hyper parameters for the model.
    kind: (optional) The kind of model to build.  Training vs inference require
      different graphs.
    datasets: The datasets structure (see top of lfads.py).

  Returns:
    an LFADS model.
  """

  build_kind = kind
  if build_kind == "write_model_params":
    build_kind = "train"
  with tf.variable_scope("LFADS", reuse=None):
    model = LFADS(hps, kind=build_kind, datasets=datasets)

  if not os.path.exists(hps.lfads_save_dir):
    print("Save directory %s does not exist, creating it." % hps.lfads_save_dir)
    os.makedirs(hps.lfads_save_dir)

  cp_pb_ln = hps.checkpoint_pb_load_name
  cp_pb_ln = 'checkpoint' if cp_pb_ln == "" else cp_pb_ln
  if cp_pb_ln == 'checkpoint':
    print("Loading latest training checkpoint in: ", hps.lfads_save_dir)
    saver = model.seso_saver
  elif cp_pb_ln == 'checkpoint_lve':
    print("Loading lowest validation checkpoint in: ", hps.lfads_save_dir)
    saver = model.lve_saver
  else:
    print("Loading checkpoint: ", cp_pb_ln, ", in: ", hps.lfads_save_dir)
    saver = model.seso_saver

  ckpt = tf.train.get_checkpoint_state(hps.lfads_save_dir,
                                       latest_filename=cp_pb_ln)

  session = tf.get_default_session()
  print("ckpt: ", ckpt)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    if kind in ["posterior_sample_and_average", "posterior_push_mean",
                "prior_sample", "write_model_params"]:
      print("Possible error!!! You are running ", kind, " on a newly \
      initialized model!")
      # cannot print ckpt.model_check_point path if no ckpt
      print("Are you sure you sure a checkpoint in ", hps.lfads_save_dir,
            " exists?")

    tf.global_variables_initializer().run()

  if ckpt:
    train_step_str = re.search('-[0-9]+$', ckpt.model_checkpoint_path).group()
  else:
    train_step_str = '-0'

  fname = 'hyperparameters' + train_step_str + '.txt'
  hp_fname = os.path.join(hps.lfads_save_dir, fname)
  hps_for_saving = jsonify_dict(hps)
  utils.write_data(hp_fname, hps_for_saving, use_json=True)

  return model


def jsonify_dict(d):
  """Turns python booleans into strings so hps dict can be written in json.
  Creates a shallow-copied dictionary first, then accomplishes string
  conversion.

  Args:
    d: hyperparameter dictionary

  Returns: hyperparameter dictionary with bool's as strings
  """

  d2 = d.copy()   # shallow copy is fine by assumption of d being shallow
  def jsonify_bool(boolean_value):
    if boolean_value:
      return "true"
    else:
      return "false"

  for key in d2.keys():
    if isinstance(d2[key], bool):
      d2[key] = jsonify_bool(d2[key])
  return d2


def build_hyperparameter_dict(flags):
  """Simple script for saving hyper parameters.  Under the hood the
  flags structure isn't a dictionary, so it has to be simplified since we
  want to be able to view file as text.

  Args:
    flags: From tf.app.flags

  Returns:
    dictionary of hyper parameters (ignoring other flag types).
  """
  d = {}
  # Data
  d['output_dist'] = flags.output_dist
  d['data_dir'] = flags.data_dir
  d['lfads_save_dir'] = flags.lfads_save_dir
  d['checkpoint_pb_load_name'] = flags.checkpoint_pb_load_name
  d['checkpoint_name'] = flags.checkpoint_name
  d['output_filename_stem'] = flags.output_filename_stem
  d['max_ckpt_to_keep'] = flags.max_ckpt_to_keep
  d['max_ckpt_to_keep_lve'] = flags.max_ckpt_to_keep_lve
  d['ps_nexamples_to_process'] = flags.ps_nexamples_to_process
  d['ext_input_dim'] = flags.ext_input_dim
  d['data_filename_stem'] = flags.data_filename_stem
  d['device'] = flags.device
  d['csv_log'] = flags.csv_log
  d['num_steps_for_gen_ic'] = flags.num_steps_for_gen_ic
  d['inject_ext_input_to_gen'] = flags.inject_ext_input_to_gen
  # Cell
  d['cell_weight_scale'] = flags.cell_weight_scale
  # Generation
  d['ic_dim'] = flags.ic_dim
  d['factors_dim'] = flags.factors_dim
  d['ic_enc_dim'] = flags.ic_enc_dim
  d['gen_dim'] = flags.gen_dim
  d['gen_cell_input_weight_scale'] = flags.gen_cell_input_weight_scale
  d['gen_cell_rec_weight_scale'] = flags.gen_cell_rec_weight_scale
  # KL distributions
  d['ic_prior_var_min'] = flags.ic_prior_var_min
  d['ic_prior_var_scale'] = flags.ic_prior_var_scale
  d['ic_prior_var_max'] = flags.ic_prior_var_max
  d['ic_post_var_min'] = flags.ic_post_var_min
  d['co_prior_var_scale'] = flags.co_prior_var_scale
  d['prior_ar_atau'] = flags.prior_ar_atau
  d['prior_ar_nvar'] =  flags.prior_ar_nvar
  d['do_train_prior_ar_atau'] = flags.do_train_prior_ar_atau
  d['do_train_prior_ar_nvar'] = flags.do_train_prior_ar_nvar
  # Controller
  d['do_causal_controller'] = flags.do_causal_controller
  d['controller_input_lag'] = flags.controller_input_lag
  d['do_feed_factors_to_controller'] = flags.do_feed_factors_to_controller
  d['feedback_factors_or_rates'] = flags.feedback_factors_or_rates
  d['co_dim'] = flags.co_dim
  d['ci_enc_dim'] = flags.ci_enc_dim
  d['con_dim'] = flags.con_dim
  d['co_mean_corr_scale'] = flags.co_mean_corr_scale
  # Optimization
  d['batch_size'] = flags.batch_size
  d['learning_rate_init'] = flags.learning_rate_init
  d['learning_rate_decay_factor'] = flags.learning_rate_decay_factor
  d['learning_rate_stop'] = flags.learning_rate_stop
  d['learning_rate_n_to_compare'] = flags.learning_rate_n_to_compare
  d['max_grad_norm'] = flags.max_grad_norm
  d['cell_clip_value'] = flags.cell_clip_value
  d['do_train_io_only'] = flags.do_train_io_only
  d['do_train_encoder_only'] = flags.do_train_encoder_only
  d['do_reset_learning_rate'] = flags.do_reset_learning_rate
  d['do_train_readin'] = flags.do_train_readin

  # Overfitting
  d['keep_prob'] = flags.keep_prob
  d['temporal_spike_jitter_width'] = flags.temporal_spike_jitter_width
  d['l2_gen_scale'] = flags.l2_gen_scale
  d['l2_con_scale'] = flags.l2_con_scale
  # Underfitting
  d['kl_ic_weight'] = flags.kl_ic_weight
  d['kl_co_weight'] = flags.kl_co_weight
  d['kl_start_step'] = flags.kl_start_step
  d['kl_increase_steps'] = flags.kl_increase_steps
  d['l2_start_step'] = flags.l2_start_step
  d['l2_increase_steps'] = flags.l2_increase_steps

  return d


class hps_dict_to_obj(dict):
  """Helper class allowing us to access hps dictionary more easily."""

  def __getattr__(self, key):
    if key in self:
      return self[key]
    else:
      assert False, ("%s does not exist." % key)
  def __setattr__(self, key, value):
    self[key] = value


def train(hps, datasets):
  """Train the LFADS model.

  Args:
    hps: The dictionary of hyperparameters.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
  """
  model = build_model(hps, kind="train", datasets=datasets)
  if hps.do_reset_learning_rate:
    sess = tf.get_default_session()
    sess.run(model.learning_rate.initializer)

  model.train_model(datasets)


def write_model_runs(hps, datasets, output_fname=None, push_mean=False):
  """Run the model on the data in data_dict, and save the computed values.

  LFADS generates a number of outputs for each examples, and these are all
  saved.  They are:
    The mean and variance of the prior of g0.
    The mean and variance of approximate posterior of g0.
    The control inputs (if enabled)
    The initial conditions, g0, for all examples.
    The generator states for all time.
    The factors for all time.
    The rates for all time.

  Args:
    hps: The dictionary of hyperparameters.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
    output_fname (optional): output filename stem to write the model runs.
    push_mean: if False (default), generates batch_size samples for each trial
      and averages the results. if True, runs each trial once without noise,
      pushing the posterior mean initial conditions and control inputs through
      the trained model. False is used for posterior_sample_and_average, True
      is used for posterior_push_mean.
  """
  model = build_model(hps, kind=hps.kind, datasets=datasets)
  model.write_model_runs(datasets, output_fname, push_mean)


def write_model_samples(hps, datasets, dataset_name=None, output_fname=None):
  """Use the prior distribution to generate samples from the model.
  Generates batch_size number of samples (set through FLAGS).

  LFADS generates a number of outputs for each examples, and these are all
  saved.  They are:
    The mean and variance of the prior of g0.
    The control inputs (if enabled)
    The initial conditions, g0, for all examples.
    The generator states for all time.
    The factors for all time.
    The output distribution parameters (e.g. rates) for all time.

  Args:
    hps: The dictionary of hyperparameters.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
    dataset_name: The name of the dataset to grab the factors -> rates
      alignment matrices from. Only a concern with models trained on
      multi-session data. By default, uses the first dataset in the data dict.
    output_fname: The name prefix of the file in which to save the generated
      samples.
  """
  if not output_fname:
    output_fname = "model_runs_" + hps.kind
  else:
    output_fname = output_fname + "model_runs_" + hps.kind
  if not dataset_name:
    dataset_name = datasets.keys()[0]
  else:
    if dataset_name not in datasets.keys():
      raise ValueError("Invalid dataset name '%s'."%(dataset_name))
  model = build_model(hps, kind=hps.kind, datasets=datasets)
  model.write_model_samples(dataset_name, output_fname)


def write_model_parameters(hps, output_fname=None, datasets=None):
  """Save all the model parameters

  Save all the parameters to hps.lfads_save_dir.

  Args:
    hps: The dictionary of hyperparameters.
    output_fname: The prefix of the file in which to save the generated
      samples.
    datasets: A dictionary of data dictionaries.  The dataset dict is simply a
      name(string)-> data dictionary mapping (See top of lfads.py).
  """
  if not output_fname:
    output_fname = "model_params"
  else:
    output_fname = output_fname + "_model_params"
  fname = os.path.join(hps.lfads_save_dir, output_fname)
  print("Writing model parameters to: ", fname)
  # save the optimizer params as well
  model = build_model(hps, kind="write_model_params", datasets=datasets)
  model_params = model.eval_model_parameters(use_nested=False,
                                             include_strs="LFADS")
  utils.write_data(fname, model_params, compression=None)
  print("Done.")


def clean_data_dict(data_dict):
  """Add some key/value pairs to the data dict, if they are missing.
  Args:
    data_dict - dictionary containing data for LFADS
  Returns:
    data_dict with some keys filled in, if they are absent.
  """

  keys = ['train_truth', 'train_ext_input', 'valid_data',
          'valid_truth', 'valid_ext_input', 'valid_train']
  for k in keys:
    if k not in data_dict:
      data_dict[k] = None

  return data_dict


def load_datasets(data_dir, data_filename_stem):
  """Load the datasets from a specified directory.

  Example files look like
    >data_dir/my_dataset_first_day
    >data_dir/my_dataset_second_day

  If my_dataset (filename) stem is in the directory, the read routine will try
  and load it.  The datasets dictionary will then look like
  dataset['first_day'] -> (first day data dictionary)
  dataset['second_day'] -> (first day data dictionary)

  Args:
    data_dir: The directory from which to load the datasets.
    data_filename_stem: The stem of the filename for the datasets.

  Returns:
    datasets: a dataset dictionary, with one name->data dictionary pair for
    each dataset file.
  """
  print("Reading data from ", data_dir)
  datasets = utils.read_datasets(data_dir, data_filename_stem)
  for k, data_dict in datasets.items():
    datasets[k] = clean_data_dict(data_dict)

    train_total_size = len(data_dict['train_data'])
    if train_total_size == 0:
      print("Did not load training set.")
    else:
      print("Found training set with number examples: ", train_total_size)

    valid_total_size = len(data_dict['valid_data'])
    if valid_total_size == 0:
      print("Did not load validation set.")
    else:
      print("Found validation set with number examples: ", valid_total_size)

  return datasets
