from __future__ import print_function, division
from builtins import range
import numpy as np

from cs231n.layers import affine_forward, affine_backward, softmax_loss_mask, softmax_loss

def rnn_step_forward(x, prev_h, Wx, Wh, bh):
    current_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + bh)
    cache = (x, prev_h, Wx, Wh, bh, current_h)
    return current_h, cache

def rnn_step_backward(dcurrent_h, cache):
    (x, prev_h, Wx, Wh, bh, current_h) = cache

    dcurrent_state = dcurrent_h * (1 - np.square(current_h))
    dx = dcurrent_state.dot(Wx.T)
    dWx = x.T.dot(dcurrent_state)
    dprev_h = dcurrent_state.dot(Wh.T)
    dWh = prev_h.T.dot(dcurrent_state)
    dbh = np.sum(dcurrent_state, axis=0)
    return dx, dprev_h, dWx, dWh, dbh

def temporal_affine_forward(h, w, b):
    (N, T, H) = h.shape
    (C, ) = b.shape
    
    out = h.reshape(N*T, H).dot(w) + b
    out = out.reshape(N, T, C)
    cache = (h, w, b, out)
    return out, cache

def temporal_affine_backward(dout, cache):
    (h, w, b, out) = cache
    (N, T, H) = h.shape
    (C, ) = b.shape    
    
    dout = dout.reshape(N*T, C)
    dh = dout.dot(w.T).reshape(N, T, H)
    dw = h.reshape(N*T, H).T.dot(dout)
    db = np.sum(dout, axis=0)
    
    return dh, dw, db

def temporal_softmax_loss(x, y, mask):
    (N, T, C) = x.shape
    x_flat = x.reshape(N*T, C)
    y_flat = y.reshape(N*T)
    mask_flat = mask.reshape(N*T)
    
    probs = np.exp(x_flat - np.max(x_flat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    loss = -np.sum(mask_flat * np.log(probs[np.arange(N*T), y_flat])) / N
    dx_flat = probs.copy()
    dx_flat[np.arange(N*T), y_flat] -= 1
    dx_flat = dx_flat * mask_flat[:, np.newaxis]
    dx_flat /= N
    dx = dx_flat.reshape(N, T, C)
    return loss, dx  

def rnn_forward(x, h0, Wx, Wh, bh):
    (N, T, W) = x.shape
    H = bh.shape[0]
    
    h = np.zeros((N, T, H))
    caches = {}
    caches['T'] = T
    
    prev_h = h0
    for i in range(T):
        current_x = x[:, i, :]
        cache_name = 'cache%d' %i
        h[:, i, :], caches[cache_name] = rnn_step_forward(current_x, prev_h, Wx, Wh, bh)
        prev_h = h[:, i, :]
        
    return h, caches

def rnn_backward(dLossdh, caches):
    cache0 = caches['cache0']
    (x, prev_h, Wx, Wh, bh, current_h) = cache0
    (N, W) = x.shape
    T = caches['T']
    H = bh.shape[0]
    
    dx = np.zeros((N, T, W))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dbh = np.zeros_like(bh)
    dprev_h = np.zeros_like(prev_h)
    for i in reversed(range(T)):
        cache_name = 'cache%d' %i
        cache = caches[cache_name]
        dcurrent_h = dLossdh[:, i, :] + dprev_h
        dx[:, i, :], dprev_h, dWx_, dWh_, dbh_ = rnn_step_backward(dcurrent_h, cache)
        dWx += dWx_
        dWh += dWh_
        dbh += dbh_
    dh0 = dprev_h
    return dx, dh0, dWx, dWh, dbh

def rnn_step_full_forward(x, prev_h, Wx, Wh, bh, Ws, bs, y, mask):
    current_h = np.tanh(x.dot(Wx) + prev_h.dot(Wh) + bh)
    current_score, cache_affine = affine_forward(current_h, Ws, bs)
    current_loss, dscore = softmax_loss_mask(current_score, y, mask)
#     current_loss, dscore = softmax_loss(current_score, y)
    cache = (x, prev_h, Wx, Wh, bh, Ws, bs, current_h, cache_affine, dscore)
    return current_h, current_loss, cache

def rnn_step_full_backward(dcurrent_h, cache):
    (x, prev_h, Wx, Wh, bh, Ws, bs, current_h, cache_affine, dscore) = cache
    
    dcurrent_h_, dWs, dbs = affine_backward(dscore, cache_affine)
    
    dcurrent_h = dcurrent_h + dcurrent_h_
    dcurrent_state = dcurrent_h * (1 - np.square(current_h))
  
    dx = dcurrent_state.dot(Wx.T)
    dWx = x.T.dot(dcurrent_state)
    dprev_h = dcurrent_state.dot(Wh.T)
    dWh = prev_h.T.dot(dcurrent_state)
    dbh = np.sum(dcurrent_state, axis=0)
    return dx, dprev_h, dWx, dWh, dbh, dWs, dbs

def rnn_full_forward(x, h0, Wx, Wh, bh, Ws, bs, y, masks):
    (N, T, W) = x.shape
    H = bh.shape[0]
    C = bs.shape[0]
    
    loss = 0.0
    h = np.zeros((N, T, H))
    caches = {}
    caches['T'] = T
    
    prev_h = h0
    for i in range(T):
        current_x = x[:, i, :]
        current_y = y[:, i]
        current_mask = masks[:, i]
        cache_name = 'cache%d' %i
        h[:, i, :], current_loss, caches[cache_name] = \
                                        rnn_step_full_forward(current_x, prev_h, Wx, Wh, bh, Ws, bs, current_y, current_mask)
        prev_h = h[:, i, :]
        loss += current_loss
    return loss, h, caches
    
def rnn_full_backward(caches):
    cache0 = caches['cache0']
    (x, prev_h, Wx, Wh, bh, Ws, bs, current_h, cache_affine, dscore) = cache0
    (N, W) = x.shape
    T = caches['T']
    H = bh.shape[0]
    C = bs.shape[0]
    
    dx = np.zeros((N, T, W))
    dWx = np.zeros_like(Wx)
    dWh = np.zeros_like(Wh)
    dbh = np.zeros_like(bh)
    dWs = np.zeros_like(Ws)
    dbs = np.zeros_like(bs)
    dprev_h = np.zeros_like(prev_h)
    for i in reversed(range(T)):
        cache_name = 'cache%d' %i
        cache = caches[cache_name]
        dcurrent_h = dprev_h
        
        dx[:, i, :], dprev_h, dWx_, dWh_, dbh_, dWs_, dbs_ = rnn_step_full_backward(dcurrent_h, cache)

        dWx += dWx_
        dWh += dWh_
        dbh += dbh_
        dWs += dWs_
        dbs += dbs_
    dh0 = dprev_h
    return dx, dh0, dWx, dWh, dbh, dWs, dbs
    
def word_embedding_forward(x, W):
    
#     N, T = x.shape
#     V, D = W.shape
#     out = np.zeros((N, T, D))
#     for n in range(N):
#         for t in range(T):
#             out[n, t, :] = W[x[n, t], :]

    out = W[x, :]
    cache = (x, W)
    return out, cache

def word_embedding_backward(dout, cache):
    (x, W) = cache
    
#     N, T = x.shape
#     V, D = W.shape
#     dW = np.zeros_like(W)
    
#     for v in range(V):
#         for n in range(N):
#             for t in range(T):
#                 if x[n, t] == v:
#                     dW[x[n, t], :] += dout[n, t, :]
    
    dW = np.zeros_like(W)
    np.add.at(dW, x, dout)
    return dW

def sigmoid(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)

def de_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def de_tanh(x):
    return 1 - np.square(np.tanh(x))

def lstm_step_forward(x, prev_h, prev_c, Wx, Wh, bh):
    N, H = prev_h.shape
    temp = x.dot(Wx) + prev_h.dot(Wh) + bh
    
    input_gate = sigmoid(temp[:, :H])
    forget_gate = sigmoid(temp[:, H:2*H])
    output_gate = sigmoid(temp[:, 2*H:3*H])
    gate_gate = np.tanh(temp[:, 3*H:4*H])

    current_c = forget_gate * prev_c + input_gate * gate_gate
    current_h = output_gate * np.tanh(current_c)
    
    cache = (x, prev_h, prev_c, Wx, Wh, bh, input_gate, forget_gate, output_gate, gate_gate, temp, current_c)
    return current_h, current_c, cache

def lstm_step_backward(dcurrent_h, dcurrent_c, cache):
    (x, prev_h, prev_c, Wx, Wh, bh, input_gate, forget_gate, output_gate, gate_gate, temp, current_c) = cache
    N, H = prev_h.shape
    Dtemp = np.zeros_like(temp)
    
    # current_h = output_gate * np.tanh(current_c)
    Doutput_gate = dcurrent_h * np.tanh(current_c)
    Dcurrent_c = dcurrent_h * output_gate * de_tanh(current_c)
    Dcurrent_c +=  dcurrent_c
    
    # current_c = forget_gate * prev_c + input_gate * gate_gate
    Dforget_gate = Dcurrent_c * prev_c
    Dprev_c = Dcurrent_c * forget_gate
    Dinput_gate = Dcurrent_c * gate_gate
    Dgate_gate = Dcurrent_c * input_gate
    
    Dtemp[:, 3*H:4*H] = Dgate_gate * de_tanh(temp[:, 3*H:4*H])
    Dtemp[:, 2*H:3*H] = Doutput_gate * de_sigmoid(temp[:, 2*H:3*H])
    Dtemp[:, H:2*H] = Dforget_gate * de_sigmoid(temp[:, H:2*H])
    Dtemp[:, :H] = Dinput_gate * de_sigmoid(temp[:, :H])
    
    # temp = x.dot(Wx) + prev_h.dot(Wh) + bh
    dx = Dtemp.dot(Wx.T)
    dWx = x.T.dot(Dtemp)
    dprev_h = Dtemp.dot(Wh.T)
    dWh = prev_h.T.dot(Dtemp)
    dbh = np.sum(Dtemp, axis=0)
    dprev_c = Dprev_c
    
    return dx, dprev_h, dprev_c, dWx, dWh, dbh

def lstm_forward(x, h0, Wx, Wh, bh):
    N, T, W = x.shape
    N, H = h0.shape
    
    h = np.zeros((N, T, H))
    caches = {}
    caches['N'], caches['T'], caches['W'], caches['H'] = N, T, W, H
    prev_h = h0
    prev_c = np.zeros_like(h0)
    
    for i in range(T):
        cache_name = 'cache%d' %i
        input_x = x[:, i, :]
        prev_h, prev_c, caches[cache_name] = lstm_step_forward(input_x, prev_h, prev_c, Wx, Wh, bh)
        h[:, i, :] = prev_h
        
    return h, caches

def lstm_backward(dout, caches):
    N, T, W, H = caches['N'], caches['T'], caches['W'], caches['H']
    
    dx = np.zeros((N, T, W))
    dWx = np.zeros((W, 4*H))
    dWh = np.zeros((H, 4*H))
    dbh = np.zeros((4*H,))
    dcurrent_h = np.zeros((N, H))
    dcurrent_c = np.zeros((N, H))
    
    for i in reversed(range(T)):
        cache_name = 'cache%d' %i
        cache = caches[cache_name]
        dcurrent_h += dout[:, i, :]
        dx[:, i, :], dcurrent_h, dcurrent_c, dWx_, dWh_, dbh_ = lstm_step_backward(dcurrent_h, dcurrent_c, cache)
        
        dWx += dWx_
        dWh += dWh_
        dbh += dbh_
    dh0 = dcurrent_h
    return dx, dh0, dWx, dWh, dbh
