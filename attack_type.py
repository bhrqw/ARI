from attacks import *
import numpy as np

attackers = {
    'fgsm_delta': lambda predict, loss_fn, eps, nb_iter, eps_iter, num_classes, initial_const: GradientSignAttack_d(predict, loss_fn, eps)
}