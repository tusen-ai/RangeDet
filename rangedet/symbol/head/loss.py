import mxnet as mx


def sigmoid_bce_loss_with_logits(logits, labels, alpha, loss_scale=1.0, name=None):
    p = mx.sym.sigmoid(logits)  # p = 1 / (1 + mx.sym.exp(-logits))  # sigmoid
    mask_logits_GE_zero = mx.sym.broadcast_greater_equal(lhs=logits, rhs=mx.sym.zeros((1,)))  # logits>=0
    minus_logits_mask = -1. * logits * mask_logits_GE_zero  # -1 * logits * [logits>=0]
    negative_abs_logits = logits - 2 * logits * mask_logits_GE_zero  # logtis - 2 * logits * [logits>=0]
    log_one_exp_minus_abs = mx.sym.Activation(negative_abs_logits,
                                              act_type='softrelu')
    # log_one_exp_minus_abs = mx.sym.log(1. + mx.sym.exp(negative_abs_logits))
    minus_log = minus_logits_mask - log_one_exp_minus_abs
    alpha_one_p_gamma_labels = alpha * labels
    log_p_clip = mx.sym.log(mx.sym.clip(p, a_min=1e-6, a_max=1 - (1e-6)))
    one_alpha_p_gamma_one_labels = (1.0 - alpha) * (1 - labels)
    forward_term1 = alpha_one_p_gamma_labels * log_p_clip
    forward_term2 = one_alpha_p_gamma_one_labels * minus_log
    loss = -1 * loss_scale * (forward_term1 + forward_term2)
    return loss


def vari_focal_loss(pred, score, loss_scale, alpha=1., gamma=2.0):
    pred_sigmoid = mx.sym.sigmoid(pred)
    loss_init = sigmoid_bce_loss_with_logits(pred, score, alpha=0.5, loss_scale=loss_scale) * 2.0
    positive_mask = mx.sym.broadcast_greater(lhs=score, rhs=mx.sym.zeros((1, 1)))
    loss_positive = loss_init * score * positive_mask
    negative_mask = mx.sym.broadcast_equal(lhs=score, rhs=mx.sym.zeros((1, 1)))
    loss_negative = loss_init * alpha * mx.sym.power(mx.sym.abs(score - pred_sigmoid), gamma) * negative_mask
    loss = loss_negative + loss_positive
    return loss
