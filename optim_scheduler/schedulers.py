from ..imports import *

from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops


class CyclicLR:
    def __init__(self, step_size=500., max_lr=0.1, min_lr=0.01):
        self.step_size = step_size
        self.max_lr = max_lr
        self.min_lr = min_lr

    def cyclic_lr_schedule(self, global_step):
        learning_rate = ops.convert_to_tensor(self.min_lr, name="learning_rate")
        dtype = learning_rate.dtype
        global_step = math_ops.cast(global_step, dtype)
        step_size = math_ops.cast(self.step_size, dtype)

        double_step = math_ops.multiply(2., step_size)
        global_div_double_step = math_ops.divide(global_step, double_step)
        cycle = math_ops.floor (math_ops.add(1., global_div_double_step))

        # computing: x = abs( global_step / step_size – 2 * cycle + 1 )
        double_cycle = math_ops.multiply(2., cycle)
        global_div_step = math_ops.divide(global_step, step_size)
        tmp = math_ops.subtract(global_div_step, double_cycle)
        x = math_ops.abs(math_ops.add(1., tmp))

        # computing: clr = learning_rate + ( max_lr – learning_rate ) * max( 0, 1 - x )
        a1 = math_ops.maximum(0., math_ops.subtract (1., x))
        a2 = math_ops.subtract(self.max_lr, learning_rate)
        clr = math_ops.multiply(a1, a2)
        return math_ops.add(clr, learning_rate)

    def get_lr_fun(self, global_step):
        return lambda: self.cyclic_lr_schedule(global_step)


class OneCycleLR:
    def __init__(self, epochs, learning_rate, batch_size, train_len):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.batches_per_epoch = train_len//batch_size + 1
        self.current_learning_rate = 0.0

    def lr_schedule(self, t):
        self.current_learning_rate = np.interp ([t], [0, (self.epochs + 1) // 5, self.epochs], [0, self.learning_rate, 0])[0]
        return self.current_learning_rate

    def get_lr_fun(self, global_step):
        return lambda : self.lr_schedule(global_step/self.batches_per_epoch)/self.batch_size

    def get_lr(self):
        return self.current_learning_rate
