# coding=utf8

import numpy as np


class HmmForward:
    """
    hidden markov model,only test forward
    """

    def __init__(self, A, B, pi, y):
        """
        :param A: num_state * num_state, symmetric matrix
        :param B: num_state * num_observe, B(i,j) is the probability of state i to observe j
        :param pi: num_state, start hidden probability
        :param y: num_T
        make sure that sum of each row in (A, B) is 1, and sum of pi is 1.
        """
        # check A, B, pi whether is valid or not
        A_row_sum, B_row_sum, pi_sum = np.sum(A, axis=1), np.sum(B, axis=1), np.sum(pi)
        expect_row_sum = np.ones(A.shape[0])
        if np.sum(np.abs(A_row_sum - expect_row_sum)) > 1e-10 or np.sum(np.abs(B_row_sum - expect_row_sum)) > 1e-10 or \
                np.abs(pi_sum - 1) > 1e-10:
            raise Exception('make sure that sum of each row in (A, B) is 1, and sum of pi is 1.')
        self.A = np.array(A, dtype=np.float)
        self.B = np.array(B, dtype=np.float)
        self.pi = np.array(pi, dtype=np.float)
        self.y = np.array(y, dtype=np.int)
        self.num_state, self.num_observe = B.shape
        self.num_T = y.shape[0]

    def forward_observe(self):
        """
        use p(hidden state, observe) to calculate probability recursively
        :return: the probability of y
        reference: http://www.52nlp.cn/hmm-learn-best-practices-five-forward-algorithm-3
        """
        prop = np.array(self.pi, copy=True)
        prop *= self.B[:, self.y[0]]
        for i in range(1, self.num_T):
            prop = np.matmul(self.A.T, prop)
            prop *= self.B[:, self.y[i]]
        return np.sum(prop)

    def forward_hidden(self):
        """
        use p(hidden state) to calculate probability recursively
        :return: the probability of y
        this method is wrong
        for the reason that y1, y2, ..., yN are correlative
        """
        hidden_prop = np.array(self.pi, copy=True)
        y_prop = np.sum(hidden_prop * self.B[:, self.y[0]])
        for i in range(1, self.num_T):
            hidden_prop = np.matmul(self.A.T, hidden_prop)
            y_prop *= np.sum(hidden_prop * self.B[:, self.y[i]])
        return y_prop

    def forward_exhaustion(self):
        """
        exhaustion method
        sum(y state sequence|hidden state sequence) for all possible hidden state sequences
        reference:www.52nlp.cn/hmm-learn-best-practices-five-forward-algorithm-3
        :return: the probability of y
        """
        pass


if __name__ == '__main__':
    A = np.array([[0.3, 0.6, 0.1],
                  [0.1, 0.7, 0.2],
                  [0.8, 0.1, 0.1]])
    B = np.array([[0.7, 0.3],
                  [0.2, 0.8],
                  [0.4, 0.6]])
    pi = np.array([0.2, 0.6, 0.2])
    y = np.array([1, 0])
    hmm = HmmForward(A, B, pi, y)
    print('use hmm_observe, get probability is %f' % hmm.forward_observe())
    print('use hmm_hidden, get probability is %f' % hmm.forward_hidden())
    print('two methods get different result, for the reason that y1, y2 ... yN are independent, so get_hidden is wrong')
