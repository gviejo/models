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
from functions import *

MAX_INT = sys.maxsize

# Lots of hyperparameters, but most are pretty insensitive.  The
# explanation of these hyperparameters is found below, in the flags
# session.

CHECKPOINT_PB_LOAD_NAME = "checkpoint"
CHECKPOINT_NAME = "lfads_vae"
CSV_LOG = "fitlog"
OUTPUT_FILENAME_STEM = ""
DEVICE = "gpu:0" # "cpu:0", or other gpus, e.g. "gpu:1"
MAX_CKPT_TO_KEEP = 5
MAX_CKPT_TO_KEEP_LVE = 5
PS_NEXAMPLES_TO_PROCESS = MAX_INT # if larger than number of examples, process all
EXT_INPUT_DIM = 0
IC_DIM = 64
FACTORS_DIM = 20
IC_ENC_DIM = 128
GEN_DIM = 200
GEN_CELL_INPUT_WEIGHT_SCALE = 1.0
GEN_CELL_REC_WEIGHT_SCALE = 1.0
CELL_WEIGHT_SCALE = 1.0
BATCH_SIZE = 128
LEARNING_RATE_INIT = 0.01
LEARNING_RATE_DECAY_FACTOR = 0.5
LEARNING_RATE_STOP = 0.005
LEARNING_RATE_N_TO_COMPARE = 1
INJECT_EXT_INPUT_TO_GEN = False
DO_TRAIN_IO_ONLY = False
DO_TRAIN_ENCODER_ONLY = False
DO_RESET_LEARNING_RATE = False
FEEDBACK_FACTORS_OR_RATES = "factors"
DO_TRAIN_READIN = True

# Calibrated just above the average value for the rnn synthetic data.
MAX_GRAD_NORM = 200.0
CELL_CLIP_VALUE = 5.0
KEEP_PROB = 0.95
TEMPORAL_SPIKE_JITTER_WIDTH = 0
OUTPUT_DISTRIBUTION = 'poisson' # 'poisson' or 'gaussian'
NUM_STEPS_FOR_GEN_IC = MAX_INT # set to num_steps if greater than num_steps

DATA_DIR = "/tmp/rnn_synth_data_v1.0/"
DATA_FILENAME_STEM = "chaotic_rnn_no_inputs"
LFADS_SAVE_DIR = "/tmp/lfads_chaotic_rnn_no_inputs/"
CO_DIM = 0
DO_CAUSAL_CONTROLLER = False
DO_FEED_FACTORS_TO_CONTROLLER = True
CONTROLLER_INPUT_LAG = 1
PRIOR_AR_AUTOCORRELATION = 10.0
PRIOR_AR_PROCESS_VAR = 0.1
DO_TRAIN_PRIOR_AR_ATAU = True
DO_TRAIN_PRIOR_AR_NVAR = True
CI_ENC_DIM = 128
CON_DIM = 128
CO_PRIOR_VAR_SCALE = 0.1
KL_INCREASE_STEPS = 2000
L2_INCREASE_STEPS = 2000
L2_GEN_SCALE = 2000.0
L2_CON_SCALE = 0.0
# scale of regularizer on time correlation of inferred inputs
CO_MEAN_CORR_SCALE = 0.0
KL_IC_WEIGHT = 1.0
KL_CO_WEIGHT = 1.0
KL_START_STEP = 0
L2_START_STEP = 0
IC_PRIOR_VAR_MIN = 0.1
IC_PRIOR_VAR_SCALE = 0.1
IC_PRIOR_VAR_MAX = 0.1
IC_POST_VAR_MIN = 0.0001      # protection from KL blowing up

flags = tf.app.flags
flags.DEFINE_string("kind",                     "train", "Type of model to build {train, posterior_sample_and_average, posterior_push_mean, prior_sample, write_model_params")
flags.DEFINE_string("output_dist",              OUTPUT_DISTRIBUTION, "Type of output distribution, 'poisson' or 'gaussian'")
flags.DEFINE_boolean("allow_gpu_growth",        False, "If true, only allocate amount of memory needed for Session. Otherwise, use full GPU memory.")
flags.DEFINE_string("data_dir",                 DATA_DIR, "Data for training")
flags.DEFINE_string("data_filename_stem",       DATA_FILENAME_STEM, "Filename stem for data dictionaries.")
flags.DEFINE_string("lfads_save_dir",           LFADS_SAVE_DIR, "model save dir")
flags.DEFINE_string("checkpoint_pb_load_name",  CHECKPOINT_PB_LOAD_NAME, "Name of checkpoint files, use 'checkpoint_lve' for best error")
flags.DEFINE_string("checkpoint_name",          CHECKPOINT_NAME, "Name of checkpoint files (.ckpt appended)")
flags.DEFINE_string("output_filename_stem",     OUTPUT_FILENAME_STEM, "Name of output file (postfix will be added)")
flags.DEFINE_string("device",                   DEVICE, "Which device to use (default: \"gpu:0\", can also be \"cpu:0\", \"gpu:1\", etc)")
flags.DEFINE_string("csv_log",                  CSV_LOG, "Name of file to keep running log of fit likelihoods, etc (.csv appended)")
flags.DEFINE_integer("max_ckpt_to_keep",        MAX_CKPT_TO_KEEP, "Max # of checkpoints to keep (rolling)")
flags.DEFINE_integer("ps_nexamples_to_process", PS_NEXAMPLES_TO_PROCESS, "Number of examples to process for posterior sample and average (not number of samples to average over).")
flags.DEFINE_integer("max_ckpt_to_keep_lve",    MAX_CKPT_TO_KEEP_LVE, "Max # of checkpoints to keep for lowest validation error models (rolling)")
flags.DEFINE_integer("ext_input_dim",           EXT_INPUT_DIM, "Dimension of external inputs")
flags.DEFINE_integer("num_steps_for_gen_ic",    NUM_STEPS_FOR_GEN_IC, "Number of steps to train the generator initial conditon.")
flags.DEFINE_boolean("inject_ext_input_to_gen", INJECT_EXT_INPUT_TO_GEN, "Should observed inputs be input to model via encoders, or injected directly into generator?")
flags.DEFINE_float("cell_weight_scale",         CELL_WEIGHT_SCALE, "Input scaling for input weights in generator.")
flags.DEFINE_integer("ic_dim",                  IC_DIM, "Dimension of h0")
flags.DEFINE_integer("factors_dim",             FACTORS_DIM, "Number of factors from generator")
flags.DEFINE_integer("ic_enc_dim",              IC_ENC_DIM, "Cell hidden size, encoder of h0")
flags.DEFINE_integer("gen_dim",                 GEN_DIM, "Cell hidden size, generator.")
flags.DEFINE_float("gen_cell_input_weight_scale", GEN_CELL_INPUT_WEIGHT_SCALE, "Input scaling for input weights in generator.")
flags.DEFINE_float("gen_cell_rec_weight_scale", GEN_CELL_REC_WEIGHT_SCALE, "Input scaling for rec weights in generator.")
flags.DEFINE_float("ic_prior_var_min",          IC_PRIOR_VAR_MIN, "Minimum variance in posterior h0 codes.")
flags.DEFINE_float("ic_prior_var_scale",        IC_PRIOR_VAR_SCALE, "Variance of ic prior distribution")
flags.DEFINE_float("ic_prior_var_max",          IC_PRIOR_VAR_MAX, "Maximum variance of IC prior distribution.")
flags.DEFINE_float("ic_post_var_min",           IC_POST_VAR_MIN, "Minimum variance of IC posterior distribution.")
flags.DEFINE_float("co_prior_var_scale",        CO_PRIOR_VAR_SCALE, "Variance of control input prior distribution.")
flags.DEFINE_float("prior_ar_atau",             PRIOR_AR_AUTOCORRELATION, "Initial autocorrelation of AR(1) priors.")
flags.DEFINE_float("prior_ar_nvar",             PRIOR_AR_PROCESS_VAR, "Initial noise variance for AR(1) priors.")
flags.DEFINE_boolean("do_train_prior_ar_atau",  DO_TRAIN_PRIOR_AR_ATAU, "Is the value for atau an init, or the constant value?")
flags.DEFINE_boolean("do_train_prior_ar_nvar",  DO_TRAIN_PRIOR_AR_NVAR, "Is the value for noise variance an init, or the constant value?")
flags.DEFINE_integer("co_dim",                  CO_DIM, "Number of control net outputs (>0 builds that graph).")
flags.DEFINE_boolean("do_causal_controller",    DO_CAUSAL_CONTROLLER, "Restrict the controller create only causal inferred inputs?")
flags.DEFINE_boolean("do_feed_factors_to_controller", DO_FEED_FACTORS_TO_CONTROLLER, "Should factors[t-1] be input to controller at time t?")
flags.DEFINE_string("feedback_factors_or_rates", FEEDBACK_FACTORS_OR_RATES, "Feedback the factors or the rates to the controller? Acceptable values: 'factors' or 'rates'.")
flags.DEFINE_integer("controller_input_lag",    CONTROLLER_INPUT_LAG, "Time lag on the encoding to controller t-lag for forward, t+lag for reverse.")
flags.DEFINE_integer("ci_enc_dim",              CI_ENC_DIM, "Cell hidden size, encoder of control inputs")
flags.DEFINE_integer("con_dim",                 CON_DIM, "Cell hidden size, controller")
flags.DEFINE_integer("batch_size",              BATCH_SIZE, "Batch size to use during training.")
flags.DEFINE_float("learning_rate_init",        LEARNING_RATE_INIT, "Learning rate initial value")
flags.DEFINE_float("learning_rate_decay_factor", LEARNING_RATE_DECAY_FACTOR, "Learning rate decay, decay by this fraction every so often.")
flags.DEFINE_float("learning_rate_stop",        LEARNING_RATE_STOP, "The lr is adaptively reduced, stop training at this value.")
flags.DEFINE_integer("learning_rate_n_to_compare", LEARNING_RATE_N_TO_COMPARE, "Number of previous costs current cost has to be worse than, to lower learning rate.")
flags.DEFINE_float("max_grad_norm",             MAX_GRAD_NORM, "Max norm of gradient before clipping.")
flags.DEFINE_float("cell_clip_value",           CELL_CLIP_VALUE, "Max value recurrent cell can take before being clipped.")
flags.DEFINE_boolean("do_train_io_only",        DO_TRAIN_IO_ONLY, "Train only the input (readin) and output (readout) affine functions.")
flags.DEFINE_boolean("do_train_encoder_only",   DO_TRAIN_ENCODER_ONLY, "Train only the encoder weights.")
flags.DEFINE_boolean("do_reset_learning_rate",  DO_RESET_LEARNING_RATE, "Reset the learning rate to initial value.")
flags.DEFINE_boolean("do_train_readin",         DO_TRAIN_READIN, "Whether to train the readin matrices and bias vectors. False leaves them fixed at their initial values specified by the alignment matrices and vectors.")
flags.DEFINE_float("keep_prob",                 KEEP_PROB, "Dropout keep probability.")
flags.DEFINE_integer("temporal_spike_jitter_width", TEMPORAL_SPIKE_JITTER_WIDTH, "Shuffle spikes around this window.")
flags.DEFINE_float("l2_gen_scale",              L2_GEN_SCALE, "L2 regularization cost for the generator only.")
flags.DEFINE_float("l2_con_scale",              L2_CON_SCALE, "L2 regularization cost for the controller only.")
flags.DEFINE_float("co_mean_corr_scale",        CO_MEAN_CORR_SCALE, "Cost of correlation (thru time)in the means of controller output.")
flags.DEFINE_float("kl_ic_weight",              KL_IC_WEIGHT, "Strength of KL weight on initial conditions KL penatly.")
flags.DEFINE_float("kl_co_weight",              KL_CO_WEIGHT, "Strength of KL weight on controller output KL penalty.")
flags.DEFINE_integer("kl_start_step",           KL_START_STEP, "Start increasing weight after this many steps.")
flags.DEFINE_integer("kl_increase_steps",       KL_INCREASE_STEPS, "Increase weight of kl cost to avoid local minimum.")
flags.DEFINE_integer("l2_start_step",           L2_START_STEP, "Start increasing l2 weight after this many steps.")
flags.DEFINE_integer("l2_increase_steps",       L2_INCREASE_STEPS, "Increase weight of l2 cost to avoid local minimum.")

FLAGS = flags.FLAGS


"""Get this whole shindig off the ground."""
d = build_hyperparameter_dict(FLAGS)
hps = hps_dict_to_obj(d)    # hyper parameters
kind = FLAGS.kind

# Read the data, if necessary.
train_set = valid_set = None
if kind in ["train", "posterior_sample_and_average", "posterior_push_mean",
    "prior_sample", "write_model_params"]:
    datasets = load_datasets(hps.data_dir, hps.data_filename_stem)
else:
    raise ValueError('Kind {} is not supported.'.format(kind))

# infer the dataset names and dataset dimensions from the loaded files
hps.kind = kind     # needs to be added here, cuz not saved as hyperparam
hps.dataset_names = []
hps.dataset_dims = {}
for key in datasets:
    hps.dataset_names.append(key)
    hps.dataset_dims[key] = datasets[key]['data_dim']

hps.num_steps = datasets[list(datasets.keys())[0]]['num_steps']
hps.ndatasets = len(hps.dataset_names)

if hps.num_steps_for_gen_ic > hps.num_steps: hps.num_steps_for_gen_ic = hps.num_steps

# Build and run the model, for varying purposes.
config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

# sys.exit()
# TRAINING
with tf.Session(config=config).as_default():
    #BUILDING MODEL
    with tf.variable_scope("LFADS", reuse=None):
        model = LFADS(hps, kind='train', datasets=datasets)

    session = tf.get_default_session()
    tf.global_variables_initializer().run()
    
    session.run(model.learning_rate.initializer)

    #TRAINING MODEL
    model.train_model(datasets)




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
