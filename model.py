import tensorflow as tf
from functools import partial

import tensorflow.contrib.rnn as rnn
import tensorflow.contrib.layers as layers
from tensorflow.python.util import nest

from cocob import COCOB
from input_pipe import InputPipe, ModelMode

from stochastic_variables import get_random_normal_variable, ExternallyParameterisedLSTM
from stochastic_variables import gaussian_mixture_nll

GRAD_CLIP_THRESHOLD = 10

def default_init(seed):
    return layers.variance_scaling_initializer(factor=1.0,
                                               mode="FAN_AVG",
                                               uniform=True,
                                               seed=seed)

def selu(x):
    """
    SELU activation
    https://arxiv.org/abs/1706.02515
    :param x:
    :return:
    """
    with tf.name_scope('elu') as scope:
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def make_encoder(time_inputs, encoder_features_depth, is_train, hparams, seed, transpose_output=True):
    """
    Builds encoder
    :param time_inputs: Input tensor, shape [batch, time, features]
    :param encoder_features_depth: Static size for features dimension
    :param is_train:
    :param hparams:
    :param seed:
    :param transpose_output: Transform RNN output to batch-first shape
    :return:
    """

    def build_cell(idx):
        with tf.variable_scope('encoder_cell', initializer=default_init(idx + seed)):
            cell = rnn.GRUCell(hparams.rnn_depth)
            has_dropout = hparams.encoder_input_dropout[idx] < 1 \
                          or hparams.encoder_state_dropout[idx] < 1 or hparams.encoder_output_dropout[idx] < 1

            if is_train and has_dropout:
                input_size = encoder_features_depth if idx == 0 else hparams.rnn_depth
                cell = rnn.DropoutWrapper(cell, dtype=tf.float32, input_size=input_size,
                                          variational_recurrent=hparams.encoder_variational_dropout[idx],
                                          input_keep_prob=1,
                                          output_keep_prob=hparams.encoder_output_dropout[idx],
                                          state_keep_prob=hparams.encoder_state_dropout[idx], seed=seed + idx)
            return cell
    if hparams.encoder_rnn_layers > 1:
        cells = [build_cell(idx) for idx in range(hparams.encoder_rnn_layers)]
        cell = rnn.MultiRNNCell(cells)
    else:
        cell = build_cell(0)
    
    def build_init_state():
        batch_len = tf.shape(time_inputs)[0]
        if hparams.encoder_rnn_layers > 1:
            return tuple([tf.zeros([batch_len, hparams.rnn_depth]) for i in range(hparams.encoder_rnn_layers)])
        else:
            return tf.zeros([batch_len, hparams.rnn_depth])
    
    # [batch, time, features] -> [time, batch, features]
    rnn_out, rnn_state = tf.nn.dynamic_rnn(cell=cell, inputs=time_inputs, dtype=tf.float32, initial_state=build_init_state())

    if transpose_output:
        rnn_out = tf.transpose(rnn_out, [1, 0, 2])
    return rnn_out, rnn_state

def calc_smape_rounded(true, predicted, weights):
    """
    Calculates SMAPE on rounded submission values. Should be close to official SMAPE in competition
    :param true:
    :param predicted:
    :param weights: Weights mask to exclude some values
    :return:
    """
    n_valid = tf.reduce_sum(weights)
    true_o = tf.round(tf.expm1(true))
    pred_o = tf.maximum(tf.round(tf.expm1(predicted)), 0.0)
    summ = tf.abs(true_o) + tf.abs(pred_o)
    zeros = summ < 0.01
    raw_smape = tf.abs(pred_o - true_o) / summ * 2.0
    smape = tf.where(zeros, tf.zeros_like(summ, dtype=tf.float32), raw_smape)
    return tf.reduce_sum(smape * weights) / n_valid


def smape_loss(true, predicted, weights):
    """
    Differentiable SMAPE loss
    :param true: Truth values
    :param predicted: Predicted values
    :param weights: Weights mask to exclude some values
    :return:
    """
    epsilon = 0.1  # Smoothing factor, helps SMAPE to be well-behaved near zero
    true_o = tf.expm1(true)
    pred_o = tf.expm1(predicted)
    summ = tf.maximum(tf.abs(true_o) + tf.abs(pred_o) + epsilon, 0.5 + epsilon)
    smape = tf.abs(pred_o - true_o) / summ * 2.0
    return tf.losses.compute_weighted_loss(smape, weights, loss_collection=None)


def decode_predictions(decoder_readout, inp: InputPipe):
    """
    Converts normalized prediction values to log1p(pageviews), e.g. reverts normalization
    :param decoder_readout: Decoder output, shape [n_days, batch]
    :param inp: Input tensors
    :return:
    """
    # [n_time, batch] -> [batch, n_time]
    batch_readout = tf.transpose(decoder_readout)
    batch_std = tf.expand_dims(inp.norm_std, -1)
    batch_mean = tf.expand_dims(inp.norm_mean, -1)
    return batch_readout * batch_std + batch_mean


def calc_loss(predictions, true_y, additional_mask=None):
    """
    Calculates losses, ignoring NaN true values (assigning zero loss to them)
    :param predictions: Predicted values
    :param true_y: True values
    :param additional_mask:
    :return: MAE loss, differentiable SMAPE loss, competition SMAPE loss
    """
    # Take into account NaN's in true values
    mask = tf.is_finite(true_y)
    # Fill NaNs by zeros (can use any value)
    true_y = tf.where(mask, true_y, tf.zeros_like(true_y))
    # Assign zero weight to NaNs
    weights = tf.to_float(mask)
    if additional_mask is not None:
        weights = weights * tf.expand_dims(additional_mask, axis=0)

    mae_loss = tf.losses.absolute_difference(labels=true_y, predictions=predictions, weights=weights)
    return mae_loss, smape_loss(true_y, predictions, weights), calc_smape_rounded(true_y, predictions,
                                                                                  weights), tf.size(true_y)

def make_train_op(loss, ema_decay=None, prefix=None):
    optimizer = COCOB()
    glob_step = tf.train.get_global_step()
    # Add regularization losses
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = loss + reg_losses if reg_losses else loss

    # Clip gradients
    grads_and_vars = optimizer.compute_gradients(total_loss)
    gradients, variables = zip(*grads_and_vars)
    clipped_gradients, glob_norm = tf.clip_by_global_norm(gradients, GRAD_CLIP_THRESHOLD)
    sgd_op, glob_norm = optimizer.apply_gradients(zip(clipped_gradients, variables)), glob_norm

    # Apply SGD averaging
    if ema_decay:
        ema = tf.train.ExponentialMovingAverage(decay=ema_decay, num_updates=glob_step)
        if prefix:
            # Some magic to handle multiple models trained in single graph
            ema_vars = [var for var in variables if var.name.startswith(prefix)]
        else:
            ema_vars = variables
        update_ema = ema.apply(ema_vars)
        with tf.control_dependencies([sgd_op]):
            training_op = tf.group(update_ema)
    else:
        training_op = sgd_op
        ema = None
    return training_op, glob_norm, ema

def rnn_stability_loss(rnn_output, beta):
    """
    REGULARIZING RNNS BY STABILIZING ACTIVATIONS
    https://arxiv.org/pdf/1511.08400.pdf
    :param rnn_output: [time, batch, features]
    :return: loss value
    """
    if beta == 0.0:
        return 0.0
    # [time, batch, features] -> [time, batch]
    l2 = tf.sqrt(tf.reduce_sum(tf.square(rnn_output), axis=-1))
    #  [time, batch] -> []
    return beta * tf.reduce_mean(tf.square(l2[1:] - l2[:-1]))


def rnn_activation_loss(rnn_output, beta):
    """
    REGULARIZING RNNS BY STABILIZING ACTIVATIONS
    https://arxiv.org/pdf/1511.08400.pdf
    :param rnn_output: [time, batch, features]
    :return: loss value
    """
    if beta == 0.0:
        return 0.0
    return tf.nn.l2_loss(rnn_output) * beta

def embedding(vm_size, embedding_size, vm_id, seed):
    # Map vm_ix to an integer
    with tf.variable_scope('embedding', initializer=default_init(seed), reuse=tf.AUTO_REUSE):
        embeddings = tf.get_variable('fc1', [vm_size, embedding_size])
        embed = tf.nn.embedding_lookup(embeddings, vm_id)
        embed = layers.batch_norm(selu(embed))
    return embed
        
class Model:
    def __init__(self, inp: InputPipe, hparams, is_train, seed, graph_prefix=None, asgd_decay=None, loss_mask=None):
        """
        Encoder-decoder prediction model
        :param inp: Input tensors
        :param hparams:
        :param is_train:
        :param seed:
        :param graph_prefix: Subgraph prefix for multi-model graph
        :param asgd_decay: Decay for SGD averaging
        :param loss_mask: Additional mask for losses calculation (one value for each prediction day), shape=[predict_window]
        """
        self.is_train = is_train
        self.inp = inp
        self.hparams = hparams
        self.seed = seed
        self.inp = inp

        # Embed vm id to a tensor
        vm_size = self.inp.vm_size
        self.vm_id = embedding(vm_size, hparams.embedding_size, self.inp.vm_ix, seed)
        
        #self.inp.time_x = tf.concat([self.inp.time_x, 
        #                             tf.tile(tf.expand_dims(self.vm_id, 1), [1, hparams.train_window, 1])], axis = 2)
        #self.inp.encoder_features_depth += hparams.embedding_size
        
        encoder_output, h_state = make_encoder(self.inp.time_x, self.inp.encoder_features_depth, is_train, hparams, seed,
                                                        transpose_output=False)

        # Encoder activation losses
        enc_stab_loss = rnn_stability_loss(encoder_output, hparams.encoder_stability_loss / inp.train_window)
        enc_activation_loss = rnn_activation_loss(encoder_output, hparams.encoder_activation_loss / inp.train_window)

        encoder_state = h_state
        
        
        # Run decoder
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            decoder_targets, decoder_outputs = self.decoder(encoder_state, self.inp.time_y, inp.norm_x[:, -1])
        
        # Decoder activation losses
        dec_stab_loss = rnn_stability_loss(decoder_outputs, hparams.decoder_stability_loss / inp.predict_window)
        dec_activation_loss = rnn_activation_loss(decoder_outputs, hparams.decoder_activation_loss / inp.predict_window)

        # Get final denormalized predictions
        self.predictions = decode_predictions(decoder_targets, inp)

        # Calculate losses and build training op
        if inp.mode == ModelMode.PREDICT:
            # Pseudo-apply ema to get variable names later in ema.variables_to_restore()
            # This is copypaste from make_train_op()
            if asgd_decay:
                self.ema = tf.train.ExponentialMovingAverage(decay=asgd_decay)
                variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                if graph_prefix:
                    ema_vars = [var for var in variables if var.name.startswith(graph_prefix)]
                else:
                    ema_vars = variables
                self.ema.apply(ema_vars)
        else:
            self.mae, smape_loss, self.smape, self.loss_item_count = calc_loss(self.predictions, inp.true_y, additional_mask=loss_mask)

            if is_train:
                # Sum all losses
                # total_loss = smape_loss + enc_stab_loss + dec_stab_loss + enc_activation_loss + dec_activation_loss
                total_loss = self.mae
                self.train_op, self.glob_norm, self.ema = make_train_op(total_loss, asgd_decay, prefix=graph_prefix)



    def default_init(self, seed_add=0):
        return default_init(self.seed + seed_add)

    def decoder(self, encoder_state, prediction_inputs, previous_y):
        """
        :param encoder_state: shape [batch_size, encoder_rnn_depth]
        :param prediction_inputs: features for prediction days, tensor[batch_size, time, input_depth]
        :param previous_y: Last day pageviews, shape [batch_size]
        :return: decoder rnn output
        """
        self.decoder_input_size = prediction_inputs.shape[1] + 1 + hparams.embedding_size

        with tf.variable_scope('decoder_cell', initializer=self.default_init(idx)):
            # Set up stochastic GRU cell with weights drawn from q(phi) = N(phi | mu, sigma)
            logger.info("Building LSTM cell with weights drawn from q(phi) = N(phi | mu, sigma)")
            with tf.variable_scope("phi_rnn"):

                phi_w, phi_w_mean, phi_w_std = get_random_normal_variable("phi_w", 0.0, hparams.init_scale,
                                                   [self.decoder_input_size + hparams.rnn_depth,
                                                    3 * hparams.rnn_depth], dtype=tf.float32)
                phi_b, phi_b_mean, phi_b_std = get_random_normal_variable("phi_b", 0.0, hparams.init_scale,
                                                   [3 * hparams.rnn_depth], dtype=tf.float32)
                    
            with tf.variable_scope('decoder_output_proj'):
                fc_w, fc_w_mean, fc_w_std = \
                    get_random_normal_variable("fc_w", 0.0, self.init_scale,
                                               [hparams.rnn_depth, 1], dtype=tf.float32)

                fc_b, fc_b_mean, fc_b_std = \
                    get_random_normal_variable("fc_b", 0.0, self.init_scale,
                                               [1], dtype=tf.float32) 
                

             
            # Sample from posterior and assign to GRU weights
            posterior_weights = self.sharpen_posterior(inputs, phi_cell, [phi_w, phi_b], [fc_w, fc_b])
            [theta_w, theta_b] = posterior_weights[0]
            [posterior_fc_w, posterior_fc_b] = posterior_weights[1]
            [theta_w_mean, theta_b_mean] = posterior_weights[2]
            [posterior_fc_w_mean, posterior_fc_b_mean] = posterior_weights[3]
                
            with tf.variable_scope("theta_gru"):
                theta_cell = ExternallyParameterisedGRU(theta_w, theta_b, num_units=self.hidden_size)


                
        def build_cell():
            return ExternallyParameterisedGRU(phi_w, phi_b, num_units=hparams.rnn_depth)
        
        if hparams.decoder_rnn_layers > 1:
            cells = [build_cell() for idx in range(hparams.decoder_rnn_layers)]
            cell = rnn.MultiRNNCell(cells)
        else:
            cell = build_cell()

        nest.assert_same_structure(encoder_state, cell.state_size)
        predict_days = self.inp.predict_window
        assert prediction_inputs.shape[1] == predict_days
        # [batch_size, time, input_depth] -> [time, batch_size, input_depth]
        inputs_by_time = tf.transpose(prediction_inputs, [1, 0, 2])
        # Return raw outputs for RNN losses calculation
        return_raw_outputs = self.hparams.decoder_stability_loss > 0.0 or self.hparams.decoder_activation_loss > 0.0
        
        # Stop condition for decoding loop
        def cond_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
            return time < predict_days

        # FC projecting layer to get single predicted value from RNN output
        def project_output(tensor):  
            return tf.nn.bias_add(tf.mutmul(tensor, fc_w), fc_b)
            
        def loop_fn(time, prev_output, prev_state, array_targets: tf.TensorArray, array_outputs: tf.TensorArray):
            """
            Main decoder loop
            :param time: Day number
            :param prev_output: Output(prediction) from previous step
            :param prev_state: RNN state tensor from previous step
            :param array_targets: Predictions, each step will append new value to this array
            :param array_outputs: Raw RNN outputs (for regularization losses)
            :return:
            """
            # RNN inputs for current step
            features = inputs_by_time[time]
            
            # [batch, predict_window, readout_depth * n_heads] -> [batch, readout_depth * n_heads]
            # Append previous predicted value to input features
            
            next_input = tf.concat([prev_output, features, self.vm_id], axis=1)
            # next_input = tf.concat([prev_output, features], axis=1)

            # Run RNN cell
            output, state = cell(next_input, prev_state)
            # Make prediction from RNN outputs
            projected_output = project_output(output)
            # Append step results to the buffer arrays
            if return_raw_outputs:
                array_outputs = array_outputs.write(time, output)
            array_targets = array_targets.write(time, projected_output)
            # Increment time and return
            return time + 1, projected_output, state, array_targets, array_outputs

        # Initial values for loop
        loop_init = [tf.constant(0, dtype=tf.int32),
                     tf.expand_dims(previous_y, -1),
                     encoder_state,
                     tf.TensorArray(dtype=tf.float32, size=predict_days),
                     tf.TensorArray(dtype=tf.float32, size=predict_days) if return_raw_outputs else tf.constant(0)]
        # Run the loop
        _, _, _, targets_ta, outputs_ta = tf.while_loop(cond_fn, loop_fn, loop_init)

        # Get final tensors from buffer arrays
        targets = targets_ta.stack()
        # [time, batch_size, 1] -> [time, batch_size]
        targets = tf.squeeze(targets, axis=-1)
        raw_outputs = outputs_ta.stack() if return_raw_outputs else None
        return targets, raw_outputs
    
    def sharpen_posterior(self, inputs, outputs, cell, cell_weights, softmax_weights):

        """
        We want to reduce the variance of the variational posterior q(theta) in order to speed up learning.
        In order to do this, we add some information about this specific minibatch into the posterior by
        modelling q(theta| (x,y)). We're going to compute the gradient of our current GRU parameters and sample some new ones
        using a linear combination of the gradient and the current weights. Specifically, we are going to
        sample new weights theta from:
            theta ~ N(theta | phi - mu * delta, sigma*I)
        where:
            delta = gradient of -log(p(y|phi, x) with respect to phi, the weight and bias of the LSTM.
            
        :param inputs: A list of length num_steps of tensors of shape (batch_size, decoder_input_size).
                The minibatch of inputs we are sharpening the posterior around.
        :param cell: The LSTM cell initialised with the phi parameters.
        :param cell_weights: A tuple of (phi_w, phi_b), corresponding to the parameters used
                in all 3 gates of the GRU cell.
        :return theta_weights, posterior_softmax_weights: A tuple of (theta_w, theta_b)/(softmax_w, softmax_b)
                  of the same respective shape as (phi_w, phi_b)/(softmax_w, softmax_b), parameterised as a
                  linear combination of phi and delta := -log(p(y|phi, x) by sampling from:
                  theta ~ N(theta| phi - mu * delta, sigma*I),where sigma is a hyperparameter and mu is
                  a "learning rate".
        :return theta_parameters/softmax_parameters: A tuple of (theta_w_mean, theta_b_mean)/
                  (softmax_w_mean, softmax) the mean of the normal distribution used to
                  sample theta (i.e  phi - mu * delta).
        """
        mae, smape_loss, _, _ = calc_loss(self.predictions, inp.true_y, additional_mask=loss_mask)
        cost = mae

        all_weights = cell_weights + softmax_weights

        # Gradients of log(p(y | phi, x )) with respect to phi (i.e., the log likelihood).
        gradients, _ = tf.clip_by_global_norm(tf.gradients(cost, all_weights), self.max_grad_norm)
        new_weights = []
        new_parameters = []
        parameter_name_scopes = ["phi_w_sample", "phi_b_sample", "softmax_w_sample", "softmax_b_sample"]
        for (cell_weight, log_likelihood_grad, scope) in zip(all_weights, gradients, parameter_name_scopes):

            with tf.variable_scope(scope):  # We want each parameter to use different smoothing weights.
                new_hierarchical_posterior, new_posterior_mean = self.resample(cell_weight, log_likelihood_grad)

            new_weights.append(new_hierarchical_posterior)
            new_parameters.append(new_posterior_mean)

        theta_weights = new_weights[:2]
        posterior_softmax_weights = new_weights[2:]
        theta_parameters = new_parameters[:2]
        softmax_parameters = new_parameters[2:]

        return theta_weights, posterior_softmax_weights, theta_parameters, softmax_parameters

    @staticmethod
    def resample(weight, gradient):
        """
        Given parameters phi and the gradients of phi with respect to -log(p(y|phi, x),
        sample posterior weights: theta ~ N(theta | phi - mu * delta, sigma*I).
        :param weight:
        :param gradient:
        :return:
        """
        # Per parameter "learning rate" for the posterior parameterisation.
        smoothing_variable = tf.get_variable("posterior_mean_smoothing",
                                             shape=weight.get_shape(),
                                             initializer=tf.random_normal_initializer(stddev=0.01))
        # Here we are basically saying:
        # "if we had to choose another set of weights to use, they should probably be a
        # combination of what they are now and some gradient step with momentum towards
        # the loss of our objective wrt to these parameters. Plus a very little bit of noise."
        new_posterior_mean = weight - (smoothing_variable * gradient)
        new_posterior_std = 0.02 * tf.random_normal(weight.get_shape(), mean=0.0, stddev=1.0)
        new_hierarchical_posterior = new_posterior_mean + new_posterior_std

        return new_hierarchical_posterior, new_posterior_mean