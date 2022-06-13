import numpy as np


class Distance:
    def __init__(self) -> None:
        super().__init__()

    def get_euclidean_distance(self, matt1, matt2):
        dist = 0
        for i in range(np.shape(matt1)[0]):
            for j in range(np.shape(matt1)[1]):
                dist += ((np.subtract(matt2[i, j], matt1[i, j])) ** 2)
        return np.sqrt(dist)

    def get_euclidean_distance_2(self, matt1, matt2):
        dist = 0
        for i in range(np.shape(matt1)[0]):
            dist += ((np.subtract(matt2[i], matt1[i])) ** 2)
        return np.sqrt(dist)

    def get_distance_matrix(self, test_data_count, mfcc_test, mfcc_zero, mfcc_one, mfcc_two, mfcc_three,
                            mfcc_four, mfcc_five, mfcc_six, mfcc_seven, mfcc_eight, mfcc_nine):
        dist_matt = np.zeros((test_data_count, 500))
        for i in range(np.shape(dist_matt)[0]):
            row = dist_matt[i]
            for j in range(50):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_zero[j])
            for j in range(50, 100):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_one[j - 50])
            for j in range(100, 150):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_two[j - 100])
            for j in range(150, 200):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_three[j - 150])
            for j in range(200, 250):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_four[j - 200])
            for j in range(250, 300):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_five[j - 250])
            for j in range(300, 350):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_six[j - 300])
            for j in range(350, 400):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_seven[j - 350])
            for j in range(400, 450):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_eight[j - 400])
            for j in range(450, 500):
                row[j] = self.get_euclidean_distance(mfcc_test[i], mfcc_nine[j - 450])
        return dist_matt

    def get_distance_matrix_2(self, test_data_count, lpc_test, lpc_zero, lpc_one, lpc_two, lpc_three,
                              lpc_four, lpc_five, lpc_six, lpc_seven, lpc_eight, lpc_nine):
        dist_matt = np.zeros((test_data_count, 500))

        for i in range(np.shape(dist_matt)[0]):
            row = dist_matt[i]
            for j in range(50):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_zero[j])
            for j in range(50, 100):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_one[j - 50])
            for j in range(100, 150):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_two[j - 100])
            for j in range(150, 200):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_three[j - 150])
            for j in range(200, 250):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_four[j - 200])
            for j in range(250, 300):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_five[j - 250])
            for j in range(300, 350):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_six[j - 300])
            for j in range(350, 400):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_seven[j - 350])
            for j in range(400, 450):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_eight[j - 400])
            for j in range(450, 500):
                row[j] = self.get_euclidean_distance_2(lpc_test[i], lpc_nine[j - 450])

        return dist_matt
