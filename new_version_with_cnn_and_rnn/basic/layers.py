import numpy as np
import theano
import theano.tensor as T

import model_utils
from cells import AttentionLSTMCell, BasicLSTMCell, WByWLSTMCell
from theano.tensor.signal import pool


class EmbddingLayer:
    def __init__(self, feature_dim, embed_dim, id="", w2v_file="", dic_file=""):
        """
        embedding layer
        Args:
            feature_dim: vocabulary size
            input_dim: embedding dimension
            id:
            w2v_file:
            dic_file:
        """
        self.output_dim = feature_dim
        self.embed_dim = embed_dim
        self.id = id

        params = dict()
        params[id + "word_embed"] = model_utils.init_weights((feature_dim, embed_dim))

        # load pre-trained word vector for initialization
        if w2v_file != "" and dic_file != "":
            dic = {w.strip(): i for i, w in enumerate(open(dic_file))}
            sin = open(w2v_file)
            sin.readline()
            for line in sin:
                parts = line.strip().split(" ")
                w = parts[0]
                if w not in dic:
                    continue
                value = [float(_) for _ in parts[1:]]
                if len(value) != embed_dim:
                    break
                params[id + "word_embed"][dic[w], 0:embed_dim] = value[0:embed_dim]

        # normalization word vector
        for i in range(params[id + "word_embed"].shape[0]):
            base = np.sum(params[id + "word_embed"][i, :] * params[id + "word_embed"][i, :])
            base = np.sqrt(base)
            params[id + "word_embed"][i, :] /= base

        # define parameters' names
        self.param_names = {"orign": [id + "word_embed"]}
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = model_utils.share_params(self.param_names, params)
        import json
        print "Embdding Layer Build! Params = %s" % (
            json.dumps(
                {"id":id,
                 "embed_dim":embed_dim,
                 "feature_dim":feature_dim}, indent=4)
        )

    def embedding(self, input):
        """
        embedding an input word by word
        Args:
            input: a tensor.fmatrix with shape (len_idx,)
        Returns: a tensor.ftensor3 with shap (len_idx, embed_dim)
        """
        return self.params[self.id + "word_embed"][input, :]


class PredictLayer:
    def __init__(self, hidden_dim, output_dim, drop=True, id=""):
        """
        Predict words probability depends on hidden states.

        Functions:
            predict_weights_real(input), predict_idx_real(input):
                used for predict result when training is done
            predict_weights(input), predict_idx(input):
                used for training that might be predict with dropout for hidden
        Args:
            hidden_dim:
            output_dim: can be number of words or classes
            drop: weather used drop when predict
            id:
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.id = id

        params = dict()
        params[id + "full_w"] = model_utils.init_weights((hidden_dim, output_dim))
        params[id + "full_b"] = model_utils.init_weights((output_dim))

        self.param_names = {
            "orign": [id + "full_w", id + "full_b"],
        }
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = model_utils.share_params(self.param_names, params)

        if drop:
            self.rng = np.random.RandomState(3435)
        import json
        print "BasicMLPLayer Build! Params = %s" %(
            json.dumps(
                {"id":id,
                 "hidden":hidden_dim,
                 "output_dim":output_dim,
                 "drop":drop}, indent=4)
        )

    def get_l1_cost(self):
        return T.sum(T.abs(self.params[self.id + "full_w"]))

    def get_l2_cost(self):
        return T.sum(self.params[self.id + "full_w"] ** 2)

    def predict(self, input, drop=True, train=True):
        """
        Calculate predcit based on input
        Args:
            input:
                a theano.tensor.fmatrix with shape (len_idx, hidden_dim)
            drop:
                whether use dropout when make prediction
        Returns:
            weights, a theano.tensor.fmatrix with shape (len_idx, output_dim)
            predict, a theano.tensor.ivector with shap (len_idx)
        """
        params = self.params
        id = self.id
        scale = 1.0
        # drop is used for training
        if drop and train:
            srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(999999))
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p=0.5, size=(input.shape[0], input.shape[1]))
            # The cast is important because
            # int * float32 = float64 which pulls things off the gpu
            input = input * T.cast(mask, theano.config.floatX)
        elif drop and not train:
            scale *= 0.5

        weights = input.dot(params[id + "full_w"]) * scale + params[id + "full_b"]
        weights = T.nnet.softmax(weights)
        predict = T.argmax(weights, axis=1)

        return weights, predict


class MLPLayer:
    def __init__(self, hidden_dim, output_dim, drop=True, id=""):
        """
        Predict words probability depends on hidden states.

        Args:
            hidden_dim:
            output_dim: can be number of words or classes
            drop: weather used drop when predict
            id:
        """
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop = drop
        self.id = id

        params = dict()
        params[id + "full_w"] = model_utils.init_weights((hidden_dim, output_dim))
        params[id + "full_b"] = model_utils.init_weights((output_dim))

        self.param_names = {
            "orign": [id + "full_w", id + "full_b"],
        }
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = model_utils.share_params(self.param_names, params)

        if drop:
            self.rng = np.random.RandomState(3435)
        import json
        print "PredictLayer Build! Params = %s" %(
            json.dumps(
                {"id":id,
                 "hidden":hidden_dim,
                 "output_dim":output_dim,
                 "drop":drop}, indent=4)
        )

    def get_output(self, input, drop=True, train=True):
        """
        Calculate predcit based on input
        Args:
            input:
                a theano.tensor.fmatrix with shape (len_idx, hidden_dim)
            drop:
                whether use dropout when make prediction
            train:
                different strategy for dropout when training and predicting
        Returns:
            weights, a theano.tensor.fmatrix with shape (len_idx, output_dim)
            predict, a theano.tensor.ivector with shap (len_idx)
        """
        params = self.params
        id = self.id
        scale = 1.0
        # drop is used for training
        if drop and train:
            srng = theano.tensor.shared_randomstreams.RandomStreams(self.rng.randint(999999))
            # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p=0.5, size=(input.shape[0], input.shape[1]))
            # The cast is important because
            # int * float32 = float64 which pulls things off the gpu
            input = input * T.cast(mask, theano.config.floatX)
        elif drop and not train:
            scale = 0.5    # multi drop ratio

        weights = input.dot(params[id + "full_w"]) * scale + params[id + "full_b"]

        return weights


class MLPLayers:
    def __init__(self, hidden_dims=[300, 300, 300], nonlinear=T.nnet.sigmoid, drop=True, id=""):
        """

        :param input_dim:
        :param layer_shapes[embed_dim, hidden_layer, ..., output_dim]:
        :param id:
        """
        self.layer_shapes = hidden_dims
        self.id = id
        self.nonlinear = nonlinear
        # define e
        # whether used a third part embedding

        self.layers = []
        cnt = 1
        for i, o in zip(hidden_dims[:-1], hidden_dims[1:]):
            layer = MLPLayer(hidden_dim=i, output_dim=o, id=self.id + str(1000 + cnt), drop=False)
            self.layers.append(layer)
            cnt += 1

        # collects all parameters
        self.params = dict()
        self.param_names = {"orign": [], "cache": []}
        for layer in self.layers:
            self.params.update(layer.params)

        for name in ["orign", "cache"]:
            for layer in self.layers:
                self.param_names[name] += layer.param_names[name]

        import json
        print "Build MLPModel Done! Params = %s" % (json.dumps({"id":id, "layer_shapes":hidden_dims}))

    def get_output(self, input):
        """
        input with shape (batch_size, layer_shapes[0])
        :param input:
        :return: shape (batch_size, layer_shapes[-1])
        """
        next_input = input
        for layer in self.layers:
            next_input = layer.get_output(next_input, drop=False)
            # next_input = self.nonlinear(next_input)

        return next_input


class BasicEncoder:
    def __init__(self, embed_dim, hidden_dim, batch_size, id=""):
        """
        Initialize basic sequence encoder based on LSTM Cell
        Functions:
            encode(input, mask), return h, c
                input is an theano.tensor.ftensor3 with float32, with shape (max_len, batch_size, input_dim)
                mask is used to defining sentence length, mask[i, j]=1 if there is a word.
                return:
                    hidden theano.tensor.ftensor3 with float32 for input with mask, shape is (max_len, batch_size, hidden_dim)
        Args:
            embed_dim: embedding dimension
            hidden_dim: hidden dimension
            batch_size: size of a min-batch
            id: id to identify its params
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.id = id

        params = dict()
        params[id + "encoder_w_x"] = model_utils.init_weights((embed_dim, hidden_dim * 4))
        params[id + "encoder_w_h"] = model_utils.init_weights((hidden_dim, hidden_dim * 4))
        params[id + "encoder_b"] = model_utils.init_weights((hidden_dim * 4))
        # Model params
        self.param_names = {"orign": params.keys()}
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = model_utils.share_params(self.param_names, params)
        import json
        print "Encoder Build! Params = %s" % (
            json.dumps(
                {"id":id,
                 "embed_dim":embed_dim,
                 "output_dim":hidden_dim,
                 "hidden_dim":hidden_dim,
                 "batch_size":batch_size}, indent=4)
        )

    def encode(self, input, mask, init_h=None, init_c=None):
        """
        Encoding input into vectors
        Args:
            input:
                An theano.tensor.ftensor3 with float32, with shape (max_len, batch_size, input_dim)
            mask:
                A theano.tensor.imatrix used to defining sentence length, mask[i, j]=1 if there is a word.
        Returns:
            hidden, cell_states, theano.tensor.ftensor3 with float32 for input with mask, shape is (max_len, batch_size, hidden_dim)
        """
        params = self.params
        id = self.id
        hidden_dim = T.as_tensor_variable(np.asarray(self.hidden_dim, "int32"))
        # encode input sequence with basic lstm cell
        output_info = [
            dict(initial=T.zeros([self.batch_size, self.hidden_dim])),  # init h
            dict(initial=T.zeros([self.batch_size, self.hidden_dim]))   # init c
        ]

        if not init_c and not init_h:
            output_info = [
                dict(initial=init_h),
                dict(initial=init_c)
            ]

        [hidden, cell_states], updates = theano.scan(
            fn=BasicLSTMCell,
            sequences=[input, mask],
            outputs_info=[dict(initial=T.zeros([self.batch_size, self.hidden_dim])),
                          dict(initial=T.zeros([self.batch_size, self.hidden_dim]))],
            non_sequences=[
                params[id + "encoder_w_x"],
                params[id + "encoder_w_h"],
                params[id + "encoder_b"],
                hidden_dim]
        )

        return hidden, cell_states


class BidirectionalEncoder:
    def __init__(self, embed_dim, hidden_dim, batch_size, id="", shared=True):
        """
        This is an bidirectional sequential encoder base on BasicEncoder.
        With two encoders when parameter 'shared' is true, or with a single encoder instead.
        Args:
            embed_dim: embedding dimension
            hidden_dim: hidden dimension
            batch_size: size of a min-batch
            id: id to identify its params
            shared: whether use shared parameters between two encoders
        """
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.shared = shared
        self.id = id
        # there is no self parameters for BidirectionalEncoder
        self.params = dict()
        self.param_names = {"orign": [], "cache": []}
        # left encoder scans input from left to right
        self.left_encoder = BasicEncoder(
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            id=id + "left_"
        )
        self.params.update(self.left_encoder.params)
        for name in ["orign", "cache"]:
            self.param_names[name] += self.left_encoder.param_names[name]

        # right encoder scans input from right to left
        if not shared:
            self.right_encoder = BasicEncoder(embed_dim, hidden_dim, batch_size, id + "right_")
            self.params.update(self.right_encoder.params)
            for name in ["orign", "cache"]:
                self.param_names[name] += self.right_encoder.param_names[name]
        import json
        print "Bidirectional Encoder Build! Params = %s" % (
            json.dumps(
                {"id":id,
                 "embed_dim":embed_dim,
                 "output_dim":hidden_dim,
                 "hidden_dim":hidden_dim,
                 "batch_size":batch_size
                 }, indent=4)
        )

    def encode(self, input, mask):
        """
        Encoding an input into vectors
        Args:
            input: (max_len, batch_size, input_dim)
            mask: Used to defining sentence length, mask[i, j]=1 if there is a word.
        Returns:
            hidden, cell_states, shape is (max_len, batch_size, hidden_dim)
        """
        # encode from left to right
        [left_h, left_c] = self.left_encoder.encode(input, mask)
        if self.shared:
            # encode from right to left by reverse elements on axis 0 for input and mask
            [right_h, right_c] = self.left_encoder.encode(input[::-1, :, :], mask[::-1, :], left_h, left_c)
        else:
            [right_h, right_c] = self.right_encoder.encode(input[::-1, :, :], mask[::-1, :], left_h, left_c)

        # concatenate hidden by elements
        hidden = T.concatenate([left_h, right_h[::-1, :, :]], axis=2)
        cell_states = T.concatenate([left_c, right_c[::-1, :, :]], axis=2)

        return hidden, cell_states


class WbyWDecoder:
    def __init__(self, context_dim, embed_dim, hidden_dim, max_len, batch_size, id="", generate=True):
        """
        Attention based Decoder implemented By WByWLSTMCell
        The Original Paper is REASONING ABOUT ENTAILMENT WITH NEURAL ATTENTION,
        URL, http://arxiv.org/pdf/1509.06664.pdf
        Notes: No prediction on words will be calculated.
        Args:
            context_dim: context dimension from encoder
            embed_dim: embedding dimension
            hidden_dim: hidden dimension
            max_len: sentence maximum length
            batch_size:
            id:
        """
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.batch_size = batch_size
        self.id = id
        self.generate = generate

        params = dict()
        params[id + "decoder_init"] = model_utils.init_weights((hidden_dim, hidden_dim))
        params[id + "decoder_u_x"] = model_utils.init_weights((embed_dim, hidden_dim * 4))
        params[id + "decoder_u_h"] = model_utils.init_weights((hidden_dim, hidden_dim * 4))
        params[id + "decoder_u_b"] = model_utils.init_weights((hidden_dim * 4))
        params[id + "attention_w_y"] = model_utils.init_weights((context_dim, hidden_dim))
        params[id + "attention_w_h"] = model_utils.init_weights((hidden_dim, hidden_dim))
        params[id + "attention_w_r"] = model_utils.init_weights((context_dim, hidden_dim))
        params[id + "attention_w_t"] = model_utils.init_weights((context_dim, context_dim))
        params[id + "attention_w_a"] = model_utils.init_weights((hidden_dim,))

        # this is used for calculate final hidden states if used for discriminate model
        if not generate:
            params[id + "final_w_r"] = model_utils.init_weights((context_dim, hidden_dim))
            params[id + "final_w_h"] = model_utils.init_weights((hidden_dim, hidden_dim))

        # Assign paramenters' names
        self.param_names = {"orign": params.keys()}
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = model_utils.share_params(self.param_names, params)
        import json
        print "WbyWDecoder Decoder Build! Params =%s" % (
            json.dumps(
                {"id":id,
                 "context_dim":context_dim,
                 "embed_dim":embed_dim,
                 "hidden_dim":hidden_dim,
                 "batch_size":batch_size}, indent=4)
        )

    def decode(self, encoding, mask_encoding, decode_input):
        """
        Decode with supervised information, often used for training a seq2seq model
        Args:
            encoding: encoding of input, (max_len, batch, context)
            mask_encoding: used for calculate real context, (max_len, batch_size)
            decode_input: real output used for training decoder, (max_len, batch, input_dim)
        Returns:
            decoding, decoding of each step, (max_len, batch, hidden)
            attention, attention of each step, (max_len, batch, max_len)
        """
        params = self.params
        id = self.id
        hidden_dim = T.as_tensor_variable(np.asarray(self.hidden_dim, "int32"))
        # it is necessary to make 0 of hidden state that out of the real length
        context = encoding * mask_encoding.dimshuffle(0, 1, "x")
        # attention pre-calculate for recurrent, (hidden, batch, hidden)
        cwy = context.dot(params[id + "attention_w_y"])
        # (hidden, batch, max_len)
        cy = context.dimshuffle(2, 1, 0)
        # initialitial value of hidden
        init_h = encoding[-1][:, self.hidden_dim: 2 * self.hidden_dim].dot(params[id + "decoder_init"])
        # h is decode hidden state, shape is max_len * batch_size * hidden
        # a is attention vector for encode input, shape is max_len * batch * (max_len for encode)
        [h, c, r, a], updates = theano.scan(
            fn=WByWLSTMCell,
            sequences=[decode_input],
            outputs_info=[
                dict(initial=init_h),  # ph
                dict(initial=T.zeros([self.batch_size, self.hidden_dim])),  # pc
                dict(initial=T.zeros([self.batch_size, self.context_dim]))  # pr
            ],
            non_sequences=[cy, cwy,
                           params[id + "decoder_u_x"],
                           params[id + "decoder_u_h"],
                           params[id + "decoder_u_b"],
                           params[id + "attention_w_h"],
                           params[id + "attention_w_r"],
                           params[id + "attention_w_t"],
                           params[id + "attention_w_a"],
                           hidden_dim])

        if self.generate:
            return [h, c, r, a]
        else:
            # final hidden states calculated
            fh = T.tanh(r[-1].dot(params[id + "final_w_r"]) + h[-1].dot(params[id + "final_w_h"]))
            return [h, c, r, a, fh]


class AttentionDecoder:
    def __init__(self, context_dim, embed_dim, hidden_dim, batch_size, id="", generate=True):
        """
        Attention based Decoder implemented By WByWLSTMCell
        The Original Paper is NEURAL MACHINE TRANSLATION
                    BY JOINTLY LEARNING TO ALIGN AND TRANSLATE
        The difference between AttentionDecoder and WByWDecoder is
            AttentionDecoder uses contexts to calculate hidden states ""inside"" LSTM
            WByWDecoder uses contexts to calculate hidden states ""outside"" LSTM
        Notes: No prediction on words will be calculated.
        Args:
            context_dim: context dimension from encoder
            input_dim: embedding dimension
            hidden_dim: hidden dimension
            batch_size:
            id:
        """
        self.context_dim = context_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.id = id
        self.generate = generate

        params = dict()
        params[id + "decoder_init"] = model_utils.init_weights((hidden_dim, hidden_dim))
        params[id + "decoder_u_x"] = model_utils.init_weights((embed_dim, hidden_dim * 4))
        params[id + "decoder_u_h"] = model_utils.init_weights((hidden_dim, hidden_dim * 4))
        params[id + "decoder_u_c"] = model_utils.init_weights((context_dim, hidden_dim * 4))
        params[id + "decoder_u_b"] = model_utils.init_weights((hidden_dim * 4))
        params[id + "attention_w_y"] = model_utils.init_weights((context_dim, hidden_dim))
        params[id + "attention_w_h"] = model_utils.init_weights((hidden_dim, hidden_dim))
        params[id + "attention_w_a"] = model_utils.init_weights((hidden_dim,))

        # this is used for calculate final hidden states if used for discriminate model
        if not generate:
            params[id + "final_w_r"] = model_utils.init_weights((context_dim, hidden_dim))
            params[id + "final_w_h"] = model_utils.init_weights((hidden_dim, hidden_dim))

        # Assign paramenters' names
        self.param_names = {"orign": params.keys()}
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = model_utils.share_params(self.param_names, params)
        import json
        print "AttentionDecoder Decoder Build! Params = %s" % (
            json.dumps(
                {"context_dim":context_dim,
                 "embed_dim":embed_dim,
                 "hidden_dim":hidden_dim,
                 "batch_size":batch_size}, indent=4)
        )

    def decode(self, encoding, mask_encoding, decode_input):
        """
        Decode with supervised information, often used for training a seq2seq model
        Args:
            encoding: encoding of input, (max_len, batch, context)
            mask_encoding: used for calculate real context with real length of encoder input, (max_len, batch_size)
            decode_input: real output used for training decoder, (max_len, batch, input_dim)
        Returns:
            decoding, decoding of each step, (max_len, batch, hidden)
            attention, attention of each step, (max_len, batch, max_len)
        """
        params = self.params
        id = self.id
        hidden_dim = T.as_tensor_variable(np.asarray(self.hidden_dim, "int32"))

        # it is necessary to make 0 of hidden state that out of the real length
        context = encoding * mask_encoding.dimshuffle(0, 1, "x")
        # attention pre-calculate for recurrent, (hidden, batch, hidden)
        cwy = context.dot(params[id + "attention_w_y"])
        # (hidden, batch, max_len)
        cy = context.dimshuffle(2, 1, 0)
        init_h = encoding[-1][:, self.hidden_dim: 2 * self.hidden_dim].dot(params[id + "decoder_init"])

        # h is decode hidden state, shape is max_len * batch_size * hidden
        # a is attention vector for encode input, shape is max_len * batch * (max_len for encode)
        [h, c, r, a], updates = theano.scan(
            fn=AttentionLSTMCell,
            sequences=[decode_input],
            outputs_info=[
                dict(initial=init_h),  # ph
                dict(initial=T.zeros([self.batch_size, self.hidden_dim])),  # pc
                dict(initial=T.zeros([self.batch_size, self.context_dim])),  # pr
                None
            ],
            non_sequences=[cy, cwy,
                           params[id + "decoder_u_x"],
                           params[id + "decoder_u_h"],
                           params[id + "decoder_u_c"],
                           params[id + "decoder_u_b"],
                           params[id + "attention_w_h"],
                           params[id + "attention_w_a"],
                           hidden_dim])

        if self.generate:
            return [h, c, r, a]
        else:
            # final hidden states calculated
            fh = T.tanh(r[-1].dot(params[id + "final_w_r"]) + h[-1].dot(params[id + "final_w_h"]))
            return [h, c, r, a, fh]

    def gen_init(self, encoding, mask_encoding):
        """
        Generate generate initial states
        Args:
            encoding:
        Returns:
            inti_h, init_c, init_r for LSTM cell
            cwy, cy for context
        """
        init_h = encoding[-1][:, self.hidden_dim: 2 * self.hidden_dim].dot(self.params[self.id + "decoder_init"])
        init_c = T.zeros([self.batch_size, self.hidden_dim])
        init_r = T.zeros([self.batch_size, self.context_dim])

        # it is necessary to make 0 of hidden state that out of the real length
        context = encoding * mask_encoding.dimshuffle(0, 1, "x")
        # attention pre-calculate for recurrent, (hidden, batch, hidden)
        cwy = context.dot(self.params[self.id + "attention_w_y"])
        # (hidden, batch, max_len)
        cy = context.dimshuffle(2, 1, 0)

        return init_h, init_c, init_r, cy, cwy

    def gem_next(self, x, ph, pc, pr, cy, cwy):
        params = self.params
        id = self.id
        hidden_dim = T.as_tensor_variable(np.asarray(self.hidden_dim, "int32"))

        # batch, *
        h, c, r, a = AttentionLSTMCell(x, ph, pc, pr, cy, cwy,
                                       params[id + "decoder_u_x"],
                                       params[id + "decoder_u_h"],
                                       params[id + "decoder_u_c"],
                                       params[id + "decoder_u_b"],
                                       params[id + "attention_w_h"],
                                       params[id + "attention_w_a"],
                                       hidden_dim)
        return h, c, r, a


class ConvPoolLayer:
    def __init__(self, batch_size, sen_len, embed_dim, 
                    filter_size=10, 
                    filter_shape=(10, 1, 2,2), 
                    channels=1, 
                    pooling_mode="max", 
                    id=""):
        """
        define Convolutional Layers, with filter_size filters with shape (filter_shape)
        :param filter_size:
        :param filter_shape:
        :param channels:

        conv_w is a 4-D tensor with shape(filter_size, channels, filter_height, filter_width)
        conv_b is a vector with shape(filter_size)

        """
        self.batch_size=batch_size

        self.sen_len=sen_len
        self.embed_dim=embed_dim
        self.input_shape=(batch_size, channels, sen_len, embed_dim)
        self.pooling_shape=(sen_len-filter_shape[2]+1, embed_dim-filter_shape[3]+1)

        self.filter_size = filter_size
        self.filter_shape = filter_shape
  
        self.channels = channels
        self.pooling_mode = pooling_mode
        self.id = id

        params = dict()
        params[id + "conv_w"] = model_utils.init_weights(self.filter_shape)
        params[id + "conv_b"] = model_utils.init_weights((filter_size,))

        self.param_names = {
            "orign": [id + "conv_w", id + "conv_b"],
        }
        self.param_names["cache"] = ["m_" + name for name in self.param_names["orign"]]
        # create shared parameters
        self.params = model_utils.share_params(self.param_names, params)

        import json
        print "ConvLayer Build! Params = %s" % (
            json.dumps(
                {"id":id,
                 "batch_size":batch_size,
                 "sen_len":sen_len,
                 "embed_dim":embed_dim,
                 "filter_size":filter_size,
                 "filter_shape":filter_shape,
                 "channels":channels}, indent=4)
        )


    def get_output(self, input):
        """
        input is

        :param input: A 4-D tensor with shape(batch_size, channels, sen_len, embedding_size),
                      usually, embedding_size == filter_width
        :return: A 4-D tensor with shape(batch_size, filter_size, sen_len-filter_height+1, embedding_size-filter_width+1)
        """
        # usually output is a 4-D tensor with shape(batch_size, filters, sen_len-filter_height+1, 1)
        output = T.nnet.conv2d(input=input,
                      filters=self.params[self.id + "conv_w"],
                      input_shape=self.input_shape,
                      filter_shape=self.filter_shape,
                      border_mode="valid")
        #  output = output.reshape([self.batch_size, self.filter_size, self.pooling_shape[0], self.pooling_shape[1]])
        # add a bias to each filter
        output += self.params[self.id + "conv_b"].dimshuffle("x", 0, "x", "x")

        if self.pooling_mode != "average": #self.pooling_mode == "max":
            output = pool.pool_2d(input=output,
                                 ignore_border=True,
                                 ds=self.pooling_shape,
                                 st=self.pooling_shape,
                                 padding=(0, 0),    # padding shape
                                 mode="max")
            # output = theano.printing.Print("Conv Pool Out")(output)
            return output.flatten().reshape([self.batch_size, self.filter_size])
        elif self.pooling_mode == "average":
            output = pool.pool_2d(input=output,
                                 ignore_border=True,
                                 ds=self.pooling_shape,
                                 st=self.pooling_shape,
                                 padding=(0, 0),    # padding shape
                                 mode="average_inc_pad")

            return output.flatten().reshape([self.batch_size, self.filter_size])

