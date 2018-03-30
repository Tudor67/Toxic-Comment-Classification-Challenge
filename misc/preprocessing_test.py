import numpy as np

from sklearn import preprocessing


def standard_scaler_test():
	initial_x = np.asarray([[6, 3, 7, 8], [2, 9, 1, 3], [-2, 4, 9, 1]], dtype='float64')
	x_mean = np.mean(initial_x, axis=0)
	x_std = np.std(initial_x, axis=0)
	modified_x = (initial_x - x_mean) / x_std

	scaler = preprocessing.StandardScaler().fit(initial_x)
	final_x = scaler.transform(initial_x)

	print('initial_x:\n{}'.format(initial_x))
	print('mean: {}'.format(x_mean))
	print('std:  {}'.format(x_std))
	print('modified_x:\n{}'.format(modified_x))
	print('final_x:\n{}'.format(final_x))


def main():
	standard_scaler_test()


if __name__ == '__main__':
	main()