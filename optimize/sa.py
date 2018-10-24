# coding=utf8

import numpy as np
import queue
import math


class SA:
    """
    simulated annealing
    """

    def __init__(self):
        self.n_iter = 10000

    def fit(self, func, target='min_val', *param_range):
        """
        get target of function with the method SA
        :param func: function pointer
        :param target: string, 'min_val' or 'max_val'
        :param param_range:  function parameters range, ((param1_min, param1_max), (param2_min, param2_max), ...)
        :return: target of function, that is the min or max value of function
        """
        n_param = len(param_range)
        param_range = np.array(param_range, dtype=np.float)
        x = np.mean(param_range, axis=1)
        recent_ys = queue.Queue(maxsize=10)
        for i in range(self.n_iter):
            y = func(x)
            step = (param_range[:, 1] - param_range[:, 0]) / self.n_iter * (self.n_iter - i) * 0.1
            x_new = x + step * (np.random.random() - 0.5) * 2
            if (x_new > param_range[:, 0]).all() and (x_new < param_range[:, 1]).all():
                y_new = func(x_new)
                if target == 'min_val':
                    if y_new < y:
                        recent_ys.put(y_new)
                        x = x_new
                    else:
                        prop = 1 / (1 + math.exp(-(y_new - y) / i))
                        if np.random.random() < prop:
                            recent_ys.put(y_new)
                            x = x_new
                else:
                    if y_new > y:
                        recent_ys.put(y_new)
                        x = x_new
                    else:
                        prop = 1 / (1 + math.exp(-(y - y_new) / i))
                        if np.random.random() < prop:
                            recent_ys.put(y_new)
                            x = x_new
            if recent_ys.qsize() == 10:
                recent_ys_arr = np.array(recent_ys.queue)
                mean = np.mean(recent_ys_arr)
                var = np.sum((recent_ys_arr - mean) ** 2) / recent_ys_arr.shape[0]
                std = math.sqrt(var)
                if std < mean * 1e-2:
                    break
                recent_ys.queue.pop()
        recent_ys_arr = np.array(recent_ys.queue)
        mean = np.mean(recent_ys_arr)
        var = np.sum((recent_ys_arr - mean) ** 2) / recent_ys_arr.shape[0]
        std = math.sqrt(var)
        if std > mean * 1e-1:
            raise Exception('no %s in function' % target)
        return x, func(x)

