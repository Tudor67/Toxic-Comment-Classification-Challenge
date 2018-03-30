import numpy as np

from sklearn import svm

def svm_classifier(train_x, train_y, test_x):
	#classifier = svm.LinearSVC()
	classifier = svm.SVC(C=0.7)
	classifier.fit(train_x, train_y)
	pred = classifier.predict(test_x)

	return pred


def generate_data(train_size=10004, test_size=10004, dims=300):
	train_x = np.random.rand(train_size, dims)
	train_y = np.zeros(train_size)
	train_x[train_size//2:] += 5
	train_y[train_size//2:] += 1

	test_x = np.random.rand(test_size, dims)
	test_x[test_size//2:] += 4.6

	return (train_x, train_y, test_x)


def main():
	train_x = [[0, 0], [1, 1]]
	train_y = [0, 1]
	test_x = [[2., 2.], [0.2, 0.3]]

	(train_x, train_y, test_x) = generate_data(150005, 150005, 300)
	print(svm_classifier(train_x, train_y, test_x))

if __name__ == '__main__':
	main()