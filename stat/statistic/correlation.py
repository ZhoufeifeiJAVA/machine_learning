# coding=utf8

import numpy as np
from scipy import stats


class Correlation:
    def __init__(self, arr1, arr2):
        self.arr1 = np.array(arr1)
        self.arr2 = np.array(arr2)
        if arr1.shape[0] != arr2.shape[0]:
            raise Exception('two arr length must be the same')
        self.length = self.arr1.shape[0]

    def use_scipy_normalize(self):
        # ddof=1 means divide n-1, ddof=0(default) means divide n
        return stats.zscore(self.arr1, ddof=1), stats.zscore(self.arr2, ddof=1)

    def normalize_with_n(self):
        mean_1, mean_2 = np.mean(self.arr1), np.mean(self.arr2)
        var1, var2 = np.sum((self.arr1 - mean_1) ** 2) / self.length, np.sum((self.arr2 - mean_2) ** 2) / self.length
        std1, std2 = np.sqrt(var1), np.sqrt(var2)
        return (self.arr1 - mean_1) / std1, (self.arr2 - mean_2) / std2

    def normalize_with_n_1(self):
        """
        divide n-1
        :return: normalized arr1, arr2
        """
        mean_1, mean_2 = np.mean(self.arr1), np.mean(self.arr2)
        var1, var2 = np.sum((self.arr1 - mean_1) ** 2) / \
            (self.length - 1), np.sum((self.arr2 - mean_2) ** 2) / (self.length - 1)
        std1, std2 = np.sqrt(var1), np.sqrt(var2)
        return (self.arr1 - mean_1) / std1, (self.arr2 - mean_2) / std2

    def get_correlation_index(self):
        arr1, arr2 = self.normalize_with_n()
        return np.sum(arr1 * arr2) / self.length


if __name__ == '__main__':
    # test if scipy.stats divide n or n - 1 to normalize data
    arr1 = np.array([74, 76, 77, 63, 63, 61, 72], dtype=np.float)
    arr2 = np.array([84, 83, 85, 74, 75, 81, 73], dtype=np.float)
    correlation = Correlation(arr1, arr2)
    print('scipy.stats get normalize data is: \n', correlation.use_scipy_normalize())
    print('divide n to normalize data is:\n', correlation.normalize_with_n())
    print('divide n - 1 to normalize data is:\n', correlation.normalize_with_n_1())
    print('actually in scipy.stats, zscore function params ddof=1 means divide n-1, ddof=0(default) means divide n')

    # test correlation index
    print('correlation index is ', correlation.get_correlation_index())

