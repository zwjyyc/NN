import os, sys, random, argparse, time, math, gzip
import cPickle as pickle
from collections import Counter

import numpy as np
import theano
import theano.tensor as T

from nn import get_activation_by_name, create_optimization_updates, softmax
from nn import Layer, EmbeddingLayer, QueryAttentionLayer, LSTM, RCNN, StrCNN, Dropout, apply_dropout
from utils import say, load_embedding_iterator

np.set_printoptions(precision=3)

def read_corpus_doc(path):
    with open(path) as fin:
	lines = fin.readlines()

    segs = [x.strip().split('\t\t') for x in lines]
    tmp_x = [ x[3].split('<ssssss>') for x in segs]
    tmp_x = map(lambda x: filter(lambda sent: sent, x), tmp_x)
    corpus_x = map(lambda x: map(lambda sent: sent.strip().split(), x), tmp_x)
    corpus_y = map(lambda x: map(lambda rating: (float(rating) - 1) / 4, x[2].strip().split()), segs)
    corpus_user = [x[1] for x in segs]
    corpus_hotel = [x[0] for x in segs]
    return corpus_x, corpus_y, corpus_user, corpus_hotel

def create_one_batch_doc(ids, x, y, u, h):
    max_len = 0
    for iid in ids:
	for sent in x[iid]:
	    if max_len < len(sent):
		max_len = len(sent)
    batch_x = map(lambda iid: np.asarray(map(lambda sent: sent + [-1] * (max_len - len(sent)), x[iid]), dtype = np.int32).T, ids)
    batch_w_mask = map(lambda iid: np.asarray(map(lambda sent: len(sent) * [1] + [0] * (max_len - len(sent)), x[iid]), dtype = np.float32).T, ids)
    batch_w_len = map(lambda iid: np.asarray(map(lambda sent: len(sent), x[iid]), dtype = np.float32) + np.float32(1e-4), ids)
    #sentence-level input
    batch_x = reduce(lambda doc, docs: np.concatenate((doc, docs), axis = 1), batch_x)
    batch_w_mask = reduce(lambda doc, docs: np.concatenate((doc, docs), axis = 1), batch_w_mask)
    batch_w_len = reduce(lambda doc, docs: np.concatenate((doc, docs), axis = 0), batch_w_len)
    
    #review-level input
    batch_y = np.array( [ y[i][0] for i in ids ] )
    batch_ay = np.array( [ y[i][1:] for i in ids ] )
    batch_u = np.array( [u[i] for i in ids])
    batch_h = np.array( [h[i] for i in ids])
    return batch_x, batch_y, batch_ay, batch_u, batch_h, batch_w_mask, batch_w_len, max_len

def create_batches_doc(perm, x, y, u, h, batch_size):
    lst = sorted(perm, key=lambda i: len(x[i]))
    batches_x = [ ]
    batches_w_masks = []
    batches_w_lens = []
    batches_sent_maxlen = []
    batches_sent_num = []
    batches_y = []
    batches_ay = []
    batches_u = []
    batches_h = []
    size = batch_size
    ids = [ lst[0] ]
    for i in lst[1:]:
        if len(ids) < size and len(x[i]) == len(x[ids[0]]):
            ids.append(i)
        else:
            bx, by, bay, bu, bh, bm, bl, ml = create_one_batch_doc(ids, x, y, u, h)
            batches_x.append(bx)
            batches_y.append(by)
	    batches_ay.append(bay)
	    batches_u.append(bu)
	    batches_h.append(bh)
	    batches_w_masks.append(bm)
	    batches_w_lens.append(bl)
	    batches_sent_num.append(len(x[ids[0]]))
	    batches_sent_maxlen.append(ml)
            ids = [ i ]
    bx, by, bay, bu, bh, bm, bl, ml = create_one_batch_doc(ids, x, y, u, h)
    batches_x.append(bx)
    batches_y.append(by)
    batches_ay.append(bay)
    batches_u.append(bu)
    batches_h.append(bh)
    batches_w_masks.append(bm)
    batches_w_lens.append(bl)
    batches_sent_num.append(len(x[ids[0]]))
    batches_sent_maxlen.append(ml)
    # shuffle batches
    batch_perm = range(len(batches_x))
    random.shuffle(batch_perm)
    batches_x = [ batches_x[i] for i in batch_perm ]
    batches_y = [ batches_y[i] for i in batch_perm ]
    batches_ay = [ batches_ay[i] for i in batch_perm ]
    batches_u = [ batches_u[i] for i in batch_perm ]
    batches_h = [ batches_h[i] for i in batch_perm ]
    batches_w_masks = [ batches_w_masks[i] for i in batch_perm ]
    batches_w_lens = [ batches_w_lens[i] for i in batch_perm ]
    batches_sent_maxlen = [ batches_sent_maxlen[i] for i in batch_perm ]
    batches_sent_num = [ batches_sent_num[i] for i in batch_perm ]

    return batches_x, batches_y, batches_ay, batches_u, batches_h, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num

class Model:
    def __init__(self, args, embedding_layer, u_embedding_layer, h_embedding_layer, naspects):
        self.args = args
        self.embedding_layer = embedding_layer
	self.u_embedding_layer = u_embedding_layer
	self.h_embedding_layer = h_embedding_layer
        self.naspects = naspects

    def ready(self):
        args = self.args
        embedding_layer = self.embedding_layer
	u_embedding_layer = self.u_embedding_layer
	h_embedding_layer = self.h_embedding_layer
        self.n_hidden = args.hidden_dim
        self.n_in = embedding_layer.n_d
	self.d_emb = embedding_layer.n_d
        dropout = self.dropout = theano.shared(
                np.float64(args.dropout_rate).astype(theano.config.floatX)
            )

        # x is length * batch_size
        # y is batch_size
        self.x = T.imatrix('x')
	self.query = T.imatrix('query')
	self.w_masks = T.fmatrix('mask')
	self.w_lens = T.fvector('lens')
	self.s_ml = T.iscalar('sent_maxlen')
	self.s_num = T.iscalar('sent_num')
        self.usr = T.ivector('users')
	self.hol = T.ivector('hotels')
	self.y = T.fvector('y')
	self.ay = T.fmatrix('ay')	
	
        x = self.x
	query = self.query
	usr = self.usr
	hol = self.hol
        y = self.y
	ay = self.ay
	w_masks = self.w_masks
	w_lens = self.w_lens
	s_ml = self.s_ml
	s_num = self.s_num
        n_hidden = self.n_hidden
        n_in = self.n_in
	d_emb = self.d_emb

        # fetch word embeddings
        # (len * batch_size) * n_in
        slices  = embedding_layer.forward(x.ravel())
        self.slices = slices #### important for updating word embeddings
	
	slicesq = embedding_layer.forward(query.ravel())
	self.slicesq = slicesq
	
	slicesu = u_embedding_layer.forward(usr)
	slicesh = h_embedding_layer.forward(hol)
	self.slicesu = slicesu
	self.slicesh = slicesh
	slicesu_h = T.concatenate((slicesu, slicesh), axis = 1)
        # 3-d tensor, len * batch_size * n_in
        slices = slices.reshape( (x.shape[0], x.shape[1], n_in) )
	slicesq = slicesq.reshape( (query.shape[0], query.shape[1], n_in))
	# maybe other way
	slicesq = T.mean(slicesq, axis = 1)
	# stacking the feature extraction layers
        pooling = args.pooling
        depth = args.depth
        layers = self.layers = [u_embedding_layer, h_embedding_layer]
	if args.fix_emb == False:
	    layers.append(embedding_layer)
	
	prev_output = slices
        prev_output = apply_dropout(prev_output, dropout, v2=True)
        size = 0
	
	n_hidden_t = n_hidden
	if args.direction == "bi":
	    n_hidden_t = 2 * n_hidden
	
	ww_lens = []
	for i in range(self.naspects):
	    ww_lens.append(w_lens) 
	
	ww_lens = T.concatenate(ww_lens)       

	softmax_inputs = [ ]
        activation = get_activation_by_name(args.act)
        for i in range(depth):
            if args.layer.lower() == "lstm":
                layer = LSTM(
                            n_in = n_hidden_t if i > 0 else n_in,
                            n_out = n_hidden_t,
			    direction = args.direction
                        )
            elif args.layer.lower() == "strcnn":
                layer = StrCNN(
                            n_in = n_hidden_t if i > 0 else n_in,
                            n_out = n_hidden_t,
                            activation = activation,
                            decay = args.decay,
                            order = args.order,
			    direction = args.direction
                        )
            elif args.layer.lower() == "rcnn":
                layer = RCNN(
                            n_in = n_hidden_t if i > 0 else n_in,
                            n_out = n_hidden_t,
                            activation = activation,
                            order = args.order,
                            mode = args.mode,
			    direction = args.direction
                        )
	    else:
                raise Exception("unknown layer type: {}".format(args.layer))

	    #### Query-specific Attention
            layers.append(layer)
            prev_output = layer.forward_all(prev_output, masks = w_masks)
            
	    layer = QueryAttentionLayer(
                        n_q = d_emb,
                        n_d = n_hidden_t
                )
	    layers.append(layer)
	    prev_output = layer.forward_all(prev_output, slicesq, masks = w_masks)
	    prev_output = prev_output.dimshuffle(1, 0, 2, 3)
	    prev_output = prev_output.reshape((prev_output.shape[0], slicesq.shape[0] * prev_output.shape[2], n_hidden_t))       

	    if pooling:
                softmax_inputs.append(T.sum(prev_output, axis=0)) # summing over columns
            else:
		ind = T.cast(w_lens - T.ones_like(w_lens), 'int32')
		softmax_inputs.append(prev_output.dimshuffle(1, 0, 2)[T.arange(ind.shape[0]), ind])

            prev_output = apply_dropout(prev_output, dropout)
	    size += n_hidden_t
	
        # final feature representation is the concatenation of all extraction layers
        if pooling:
	    softmax_input = T.concatenate(softmax_inputs, axis = 1) / ww_lens.dimshuffle(0, 'x')
        else:
            softmax_input = T.concatenate(softmax_inputs, axis=1)
        softmax_input = apply_dropout(softmax_input, dropout, v2=True)
	
        n_in = size
	size = 0
	softmax_inputs = [ ]
	[sentlen, emblen] = T.shape(softmax_input)
	prev_output = softmax_input.reshape((slicesq.shape[0], sentlen / slicesq.shape[0], emblen))
	prev_output = prev_output.reshape((slicesq.shape[0], s_num, prev_output.shape[1] / s_num, emblen))
	prev_output = prev_output.dimshuffle(1, 0, 2, 3)
	prev_output = prev_output.reshape((prev_output.shape[0], slicesq.shape[0] * prev_output.shape[2], emblen))
	
        for i in range(depth):
            if args.layer.lower() == "lstm":
                layer = LSTM(
                            n_in = n_hidden_t if i > 0 else n_in,
                            n_out = n_hidden_t,
			    direction = args.direction
                        )
            elif args.layer.lower() == "strcnn":
                layer = StrCNN(
                            n_in = n_hidden_t if i > 0 else n_in,
                            n_out = n_hidden_t,
                            activation = activation,
                            decay = args.decay,
                            order = args.order,
			    direction = args.direction
                        )
            elif args.layer.lower() == "rcnn":
                layer = RCNN(
                            n_in = n_hidden_t if i > 0 else n_in,
                            n_out = n_hidden_t,
                            activation = activation,
                            order = args.order,
                            mode = args.mode,
			    direction = args.direction
                        )
            else:
                raise Exception("unknown layer type: {}".format(args.layer))

            layers.append(layer)
            prev_output = layer.forward_all(prev_output)
	
	    if pooling:
                softmax_inputs.append(T.sum(prev_output, axis=0)) # summing over columns
            else:
		softmax_inputs.append(prev_output[-1])
            prev_output = apply_dropout(prev_output, dropout)
            size += n_hidden_t

        	# final feature representation is the concatenation of all extraction layers
        if pooling:
            softmax_input = T.concatenate(softmax_inputs, axis = 1) / T.cast(s_num, 'float32')
	else:
            softmax_input = T.concatenate(softmax_inputs, axis=1)
        softmax_input = apply_dropout(softmax_input, dropout, v2=True)
	softmax_input = softmax_input.reshape((slicesq.shape[0], softmax_input.shape[0] / slicesq.shape[0], softmax_input.shape[1]))
	
	p_y_given_a_xs = []
	for i in range(self.naspects):
	    layers.append( Layer(
		    n_in = size,
		    n_out = 2,
		    activation = softmax,
		    has_bias = False
		) )
	    p_y_given_a_xs.append(layers[-1].forward(softmax_input[i])[:, 1])
	
	self.p_y_given_a_xs = T.stack(p_y_given_a_xs).dimshuffle(1, 0)
	
	size = n_hidden * 2
        # feed the feature repr. to the softmax output layer
        layers.append( Layer(
                n_in = size,
                n_out = self.naspects,
                activation = softmax,
                has_bias = False
        ) )
	self.p_a_given_uh = layers[-1].forward(slicesu_h)
	
	if args.fix_emb == True:
            for l,i in zip(layers[2:], range(len(layers[2:]))):
            	say("layer {}: n_in={}\tn_out={}\n".format(
                    i, l.n_in, l.n_out
            	))
	else:
            for l,i in zip(layers[3:], range(len(layers[3:]))):
                say("layer {}: n_in={}\tn_out={}\n".format(
                    i, l.n_in, l.n_out
                ))

        # unnormalized score of y given x
        self.pred = T.sum(self.p_y_given_a_xs * self.p_a_given_uh, axis = 1)
	self.l2_loss_y = T.mean((self.pred - y) ** 2)
	self.l2_loss_ay = T.mean((self.p_y_given_a_xs - ay) ** 2)
        # adding regularizations
        self.l2_sqr = None
        self.params = [ ]
        for layer in layers:
            self.params += layer.params
        for p in self.params:
            if self.l2_sqr is None:
                self.l2_sqr = args.l2_reg * T.sum(p**2)
            else:
                self.l2_sqr += args.l2_reg * T.sum(p**2)

        nparams = sum(len(x.get_value(borrow=True).ravel()) \
                        for x in self.params)
        say("total # parameters: {}\n".format(nparams))


    def save_model(self, path, args):
         # append file suffix
        if not path.endswith(".pkl.gz"):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "wb") as fout:
            pickle.dump(
                ([ x.get_value() for x in self.params ], args, self.naspects),
                fout,
                protocol = pickle.HIGHEST_PROTOCOL
            )

    def load_model(self, path):
        if not os.path.exists(path):
            if path.endswith(".pkl"):
                path += ".gz"
            else:
                path += ".pkl.gz"

        with gzip.open(path, "rb") as fin:
            param_values, args, naspects = pickle.load(fin)

        self.args = args
        self.naspects = naspects
        self.ready()
        for x,v in zip(self.params, param_values):
            x.set_value(v)

    def eval_accuracy(self, preds_y, preds_ay, golds_y, golds_ay):
	error_y = sum([ sum((4 * ps - 4 * ys))**2 for ps, ys in zip(preds_y, golds_y) ])
        error_b = sum([ (4 * p - 4 * y)**2 if y >= 0 else 0 for ps, yss in zip(preds_y, golds_ay) for p, ys in zip(ps, yss) for y in ys])
	error_ay = sum([(4 * p - 4 * y)**2 if y >= 0 else 0 for pss, yss in zip(preds_ay, golds_ay) for ps, ys in zip(pss, yss) for p, y in zip(ps, ys) ])
	
	tot_y = sum( [ len(y) for y in golds_y ] )
	tot_ay = sum([ 1 if y >= 0 else 0 for b_ay in golds_ay for ay in b_ay for y in ay])
	return math.sqrt(error_y / tot_y), math.sqrt(error_ay / tot_ay), math.sqrt(error_b / tot_ay)


    def train(self, train, dev, test, query):
        args = self.args
        trainx, trainy, trainu, trainh = train
        batch_size = args.batch

        if dev:
	    dev_batches_x, dev_batches_y, dev_batches_ay, dev_batches_u, dev_batches_h, dev_batches_w_masks, dev_batches_w_lens, dev_batches_sent_maxlen, dev_batches_sent_num = create_batches_doc(
		    range(len(dev[0])),
		    dev[0],
		    dev[1],
		    dev[2],
		    dev[3],
		    batch_size
		)

        if test:
	    test_batches_x, test_batches_y, test_batches_ay, test_batches_u, test_batches_h, test_batches_w_masks, test_batches_w_lens, test_batches_sent_maxlen, test_batches_sent_num = create_batches_doc(
		    range(len(test[0])),
		    test[0],
		    test[1],
		    test[2],
		    test[3],
		    batch_size
		)

        cost = self.l2_loss_y + self.l2_sqr

        updates, lr, gnorm = create_optimization_updates(
                cost = cost,
                params = self.params,
                lr = args.learning_rate,
                method = args.learning
            )[:3]

        train_model = theano.function(
             	inputs = [self.x, self.y, self.ay, self.usr, self.hol, self.query, self.w_masks, self.w_lens, self.s_ml, self.s_num],
             	outputs = [ cost, gnorm ],
             	updates = updates,
             	allow_input_downcast = True
        	)

        eval_acc = theano.function(
             	inputs = [self.x, self.usr, self.hol, self.query, self.w_masks, self.w_lens, self.s_ml, self.s_num],
             	outputs = [self.pred, self.p_y_given_a_xs],
             	allow_input_downcast = True
        	)

        unchanged = 0
        best_dev = 1000.0
        dropout_prob = np.float64(args.dropout_rate).astype(theano.config.floatX)

        start_time = time.time()
        eval_period = args.eval_period

        perm = range(len(trainx))

        say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")
        if args.load:
	    preds = [ eval_acc(x, u, h, query, wm, wl, sm, sn) for x, u, h, wm, wl, sm, sn in zip(dev_batches_x, dev_batches_u, dev_batches_h, dev_batches_w_masks, dev_batches_w_lens, dev_batches_sent_maxlen, dev_batches_sent_num)]
	    preds_y = [ pred[0] for pred in preds]
            preds_ay = [ pred[1] for pred in preds]
            bests_dev = self.eval_accuracy(preds_y, preds_ay, dev_batches_y, dev_batches_ay)
	    best_dev = bests_dev[0]
	    say("\tdev overall/aspect_level/aspect_level_b/best rmse=%.4f/%.4/%.4f/.4f" % (
		bests_dev[0],
		bests_dev[1],
		bests_dev[2],
		best_dev
	    ))

	for epoch in xrange(args.max_epochs):
            unchanged += 1
            if unchanged > 20: return
            train_loss = 0.0

            random.shuffle(perm)
	    
	    batches_x, batches_y, batches_ay, batches_u, batches_h, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num = create_batches_doc(perm, trainx, trainy, trainu, trainh, batch_size)
            N = len(batches_x)
            for i in xrange(N):

                if (i + 1) % 100 == 0:
                    sys.stdout.write("\r%d" % i)
                    sys.stdout.flush()

                x = batches_x[i]
                y = batches_y[i]
		
		va, grad_norm = train_model(x, y, batches_ay[i], batches_u[i], batches_h[i], query, batches_w_masks[i], batches_w_lens[i], batches_sent_maxlen[i], batches_sent_num[i])
                train_loss += va

                # debug
                if math.isnan(va):
                    print ""
                    print i-1, i
                    print x
                    print y
		    print batches_ay[i]
		    print batches_u[i]
		    print batches_h[i]
		    print batches_w_masks[i]
		    print batches_w_lens[i]
		    print batches_sent_maxlen[i]
		    print batches_sent_num[i]
                    return

                if (i == N-1) or (eval_period > 0 and (i+1) % eval_period == 0):
                    self.dropout.set_value(0.0)

                    say( "\n" )
                    say( "Epoch %.3f\tloss=%.4f\t|g|=%s  [%.2fm]\n" % (
                            epoch + (i+1)/(N+0.0),
                            train_loss / (i+1),
                            float(grad_norm),
                            (time.time()-start_time) / 60.0
                    ))
                    say(str([ "%.2f" % np.linalg.norm(x.get_value(borrow=True)) for x in self.params ])+"\n")

                    if dev:
            		preds = [ eval_acc(x, u, h, query, wm, wl, sm, sn) for x, u, h, wm, wl, sm, sn in zip(dev_batches_x, dev_batches_u, dev_batches_h, dev_batches_w_masks, dev_batches_w_lens, dev_batches_sent_maxlen, dev_batches_sent_num)]
            		
			preds_y = [ pred[0] for pred in preds]
			preds_ay = [ pred[1] for pred in preds]
			nowfs_dev = self.eval_accuracy(preds_y, preds_ay, dev_batches_y, dev_batches_ay)
            		nowf_dev = nowfs_dev[0]
            		say("\tdev overall/aspect_level/aspect_level_b/best rmse=%.4f/%.4/%.4f/%.4f\n" % (
                	    nowfs_dev[0],
                	    nowfs_dev[1],
                	    nowfs_dev[2],
                	    best_dev
            		))

                        if nowf_dev < best_dev:
                            unchanged = 0
                            best_dev = nowf_dev
                            if args.save:
                                self.save_model(args.save, args)

                        if args.test and nowf_dev == best_dev:
			    preds = [ eval_acc(x, u, h, query, wm, wl, sm, sn) for x, u, h, wm, wl, sm, sn in zip(test_batches_x, test_batches_u, test_batches_h, test_batches_w_masks, test_batches_w_lens, test_batches_sent_maxlen, test_batches_sent_num)]
                            preds_y = [ pred[0] for pred in preds]
			    preds_ay = [ pred[1] for pred in preds]
			    nowfs_test = self.eval_accuracy(preds_y, preds_ay, test_batches_y, test_batches_ay)
                            say("\ttest overall/aspect_level/aspect_level_b rmse=%.4f/%.4/%.4f\n" % (
                            	nowfs_test[0],
                            	nowfs_test[1],
                            	nowfs_test[2]
                        	))


                        if best_dev < nowf_dev - 0.05:
                            return

                    self.dropout.set_value(dropout_prob)

                    start_time = time.time()

    def evaluate_batches_doc(self, batches_x, batches_y, batches_ay, batches_u, batches_h, query, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num, eval_function):
	preds = [eval_function(x, u, h, query, wm, wl, sm, sn) for x, u, h, wm, wl, sm, sn in zip(batches_x, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num)]
	return self.eval_accuracy(preds[0], preds[1], batches_y, batches_ay)

    def evaluate_set(self, data_x, data_y, data_u, data_h, query):

        args = self.args

        
        eval_acc = theano.function(
                inputs = [self.x, self.usr, query, self.hol, sely.w_masks, self.w_lens, self.s_ml, self.s_num],
                outputs = [self.pred, self.p_y_given_a_xs],
		allow_input_downcast = True
                )

        # create batches by grouping sentences of the same length together
	
	batches_x, batches_y, batches_ay, batches_u, batches_h, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num = create_batches_doc(
                    range(len(data_x[0])),
                    data_x,
                    data_y,
		    data_u,
		    data_h,
                    args.batch
                )

        # evaluate on the data set
        dropout_prob = np.float64(args.dropout_rate).astype(theano.config.floatX)
        self.dropout.set_value(0.0)
	accuracy = self.evaluate_batches_doc(batches_x, batches_y, batches_ay, batches_u, batches_h, query, batches_w_masks, batches_w_lens, batches_sent_maxlen, batches_sent_num, eval_acc)

        self.dropout.set_value(dropout_prob)
        return accuracy

def load_doc_corpus(embedding_layer, user_embedding_layer, hotel_embedding_layer, fileIn):
    corpus_x, corpus_y, corpus_usr, corpus_hol = read_corpus_doc(fileIn)
    corpus_x = map(lambda x: map(lambda sent: embedding_layer.map_to_ids(sent).tolist(), x), corpus_x)
    corpus_usr = user_embedding_layer.map_to_ids(corpus_usr).tolist()
    corpus_hol = hotel_embedding_layer.map_to_ids(corpus_hol).tolist()
    return corpus_x, corpus_y, corpus_usr, corpus_hol

def load_lis(fileIn):
    lis = []
    with open(fileIn, 'r') as fin:
	for line in fin.readlines():
	    if line.strip():
		lis.append(line.strip())
    return lis

def main(args):
    print args

    model = None

    assert args.embedding, "Pre-trained word embeddings required."
    assert args.user_dict, "User list required."
    assert args.hotel_dict, "Hotel list required."    
    assert args.aspect_seeds, "Aspect seeds required."

    usrlis = load_lis(args.user_dict)
    say("loaded {} users.\n".format(len(usrlis)))
    hollis = load_lis(args.hotel_dict)
    say("loaded {} hotels.\n".format(len(hollis)))
    seedslis = load_lis(args.aspect_seeds)
    say("loaded {} aspect seeds\n".format(len(seedslis)))
    usrlis.append("<unk>")
    hollis.append("<unk>")

    embedding_layer = EmbeddingLayer(
                n_d = args.hidden_dim,
                vocab = [ "<unk>" ],
                embs = load_embedding_iterator(args.embedding),
		fix_init_embs = args.fix_emb
            )
    user_embedding_layer = EmbeddingLayer(
	    	n_d = args.hidden_dim,
		vocab = usrlis
	    )
    hotel_embedding_layer = EmbeddingLayer(
		n_d = args.hidden_dim,
		vocab = hollis
	    )

    aspect_keys = map(lambda x: embedding_layer.map_to_ids(x.strip().split()).tolist(), seedslis)
    if args.train:
	train_x, train_y, train_usr, train_hol = load_doc_corpus(embedding_layer, user_embedding_layer, hotel_embedding_layer, args.train)

    if args.dev:
	dev_x, dev_y, dev_usr, dev_hol = load_doc_corpus(embedding_layer, user_embedding_layer, hotel_embedding_layer, args.dev)

    if args.test:
	test_x, test_y, test_usr, test_hol = load_doc_corpus(embedding_layer, user_embedding_layer, hotel_embedding_layer, args.test)
    
    if args.train:
        model = Model(
                    args = args,
                    embedding_layer = embedding_layer,
		    u_embedding_layer = user_embedding_layer,
		    h_embedding_layer = hotel_embedding_layer,
                    naspects = len(seedslis)
            )
	if args.load:
	    print 'loading model...'
	    model.load_model(args.load)
        else:
	    model.ready()
	
	print 'training...'
        model.train(
                (train_x, train_y, train_usr, train_hol),
                (dev_x, dev_y, dev_usr, dev_hol) if args.dev else None,
                (test_x, test_y, test_usr, test_hol) if args.test else None,
		aspect_keys,
            )

    if args.load and args.test and not args.train:
        # model.args and model.nclasses will be loaded from file
        model = Model(
                    args = None,
                    embedding_layer = embedding_layer,
                    nclasses = -1
            )
        model.load_model(args.load)
        accuracy = model.evaluate_set(test_x, test_y, test_usr, test_hol, query)
	say("\ttest overall rmse=%.4f\taspect_level rmse=%.4f\taspect_level_b rmse=%.4f" % (
	    accuracy[0],
	    accuracy[1],
	    accuracy[2]
          ))



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(sys.argv[0])
    argparser.add_argument("--train",
            type = str,
            default = "",
            help = "path to training data"
        )
    argparser.add_argument("--dev",
            type = str,
            default = "",
            help = "path to development data"
        )
    argparser.add_argument("--test",
            type = str,
            default = "",
            help = "path to test data"
        )
    argparser.add_argument("--user_dict",
	    type = str,
	    default = "",
	    help = "path to user dict"
	)
    argparser.add_argument("--hotel_dict",
	    type = str,
	    default = "",
	    help = "path to hotel dict"
	)
    argparser.add_argument("--aspect_seeds",
	    type = str,
	    default = "",
	    help = "path to aspect seeds"
	)
    argparser.add_argument("--direction",
	    type = str,
	    default = "bi",
	    help = "direction: forward, bi"
	)
    argparser.add_argument("--fix_emb",
	    type = bool,
	    default = True,
	    help = "fine tune embedding: True or False"
	)
    argparser.add_argument("--hidden_dim", "-d",
            type = int,
            default = 50,
            help = "hidden dimensions"
        )
    argparser.add_argument("--decay",
            type = float,
            default = 0.5,
            help = "the decay factor of StrCNN layer"
        )
    argparser.add_argument("--learning",
            type = str,
            default = "adadelta",
            help = "learning method (sgd, adagrad, adam, ...)"
        )
    argparser.add_argument("--learning_rate",
            type = float,
            default = "0.001",
            help = "learning rate"
        )
    argparser.add_argument("--max_epochs",
            type = int,
            default = 100,
            help = "maximum # of epochs"
        )
    argparser.add_argument("--eval_period",
            type = int,
            default = -1,
            help = "evaluate on dev every period"
        )
    argparser.add_argument("--dropout_rate",
            type = float,
            default = 0,
            help = "dropout probability"
        )
    argparser.add_argument("--l2_reg",
            type = float,
            default = 0.00001
        )
    argparser.add_argument("--embedding",
            type = str,
            default = ""
        )
    argparser.add_argument("--batch",
            type = int,
            default = 32,
            help = "mini-batch size"
        )
    argparser.add_argument("--depth",
            type = int,
            default = 1,
            help = "number of feature extraction layers (min:1)"
        )
    argparser.add_argument("--order",
            type = int,
            default = 2,
            help = "when the order is k, we use up tp k-grams"
        )
    argparser.add_argument("--act",
            type = str,
            default = "relu",
            help = "activation function (none, relu, tanh, etc.)"
        )
    argparser.add_argument("--layer",
            type = str,
            default = "lstm",
            help = "type of neural net (LSTM, RCNN, StrCNN)"
        )
    argparser.add_argument("--mode",
            type = int,
            default = 1
        )
    argparser.add_argument("--save",
            type = str,
            default = "",
            help = "save model to this file"
        )
    argparser.add_argument("--load",
            type = str,
            default = "",
            help = "load model from this file"
        )
    argparser.add_argument("--pooling",
            type = int,
            default = 1,
            help = "whether to use mean pooling or take the last vector"
        )
    args = argparser.parse_args()
    main(args)


