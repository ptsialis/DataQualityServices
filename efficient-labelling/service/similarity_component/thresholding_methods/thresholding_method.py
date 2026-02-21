import numpy as np
from sklearn.metrics.pairwise import distance_metrics

def calc_one_nearest_neighbor(centroids, target_centroid, dist="euclidean"):
    dist_func = distance_metrics()[dist]
    
    # convert centroids and target into numpy arrays (is it necessary?)
    centroids_arr = np.array(centroids)
    target_arr = np.array(target_centroid).reshape(1, -1)
    
    # compute distancee from target to all reference centroids
    distances = dist_func(target_arr, centroids_arr)[0]
    nn_index = np.argmin(distances)
    nn_distance = distances[nn_index]
    
    # if only one centroid exists classify as normal (1)
    if centroids_arr.shape[0] == 1:
        return 1
    
    # find the nearest neighbors own nearest neighbor (excluding itself)
    centroids_remaining = np.delete(centroids_arr, nn_index, axis=0)
    nearest_neighbor = centroids_arr[nn_index].reshape(1, -1)
    
    nn_distances = dist_func(nearest_neighbor, centroids_remaining)[0]
    nn_nn_index = np.argmin(nn_distances)
    nn_nn_distance = nn_distances[nn_nn_index]
    
    # indicator function (indicates "anomalous" centroid)
    def indicator(nn_distance, nn_nn_distance):
        if nn_distance == 0:
            return 1 # normal
        if nn_nn_distance == 0:
            return -1 # anomalous
        ratio = nn_distance / nn_nn_distance
        return 1 if ratio <= 1 else -1
    
    return indicator(nn_distance, nn_nn_distance)

THRESHOLDING_METHODS_LOOKUP = {
    "1-NN": calc_one_nearest_neighbor,
}

class ThresholdingMethod:
    def __init__(self, thresholding_method="1-NN"):
        try:
            self.thresholding_method = THRESHOLDING_METHODS_LOOKUP[thresholding_method]
        except KeyError:
            raise ValueError("Invalid thresholding method")
    
    def threshold_centroid(self, centroids, target_centroid):
        return self.thresholding_method(centroids, target_centroid)

if __name__ == "__main__":
    # "component" test
    ref_centroids = [
        [0, 1, 1],
        [1, 0, 0],
        [0, 0, 1]
    ]
    
    target_centroid = [20, 1, 0]

    thresholding_method = ThresholdingMethod()
    res = thresholding_method.threshold_centroid(ref_centroids, target_centroid)
    
    print("Anomaly Detection Result:", res)