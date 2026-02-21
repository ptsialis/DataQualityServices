
import numpy as np

from scipy.stats import wasserstein_distance_nd
from scipy.spatial.distance import mahalanobis
from scipy.special import softmax
from scipy.stats import entropy
from scipy.linalg import det 



def calc_euclidean_distance(dist1_data, dist2_data):
    # calc centroids
    centroid1 = np.mean(dist1_data, axis=0)
    centroid2 = np.mean(dist2_data, axis=0)
    
    # calc euclidean distance
    distance = np.linalg.norm(centroid1 - centroid2)
    
    return distance


def calc_cosine_similarity(dist1_data, dist2_data):
    # calc centroids
    centroid1 = np.mean(dist1_data, axis=0)
    centroid2 = np.mean(dist2_data, axis=0)
    
    # calc cosine similarity
    dot_product = np.dot(centroid1, centroid2)
    norm1 = np.linalg.norm(centroid1)
    norm2 = np.linalg.norm(centroid2)
    
    if norm1 == 0 or norm2 == 0:
        raise ValueError("One of the centroids is a zero vector; cannot compute cosine similarity")
    
    cosine_similarity = dot_product / (norm1 * norm2)
    return cosine_similarity


def calc_mahalanobis_distance(dist1_data, dist2_data):
    # calc centroids
    centroid1 = np.mean(dist1_data, axis=0)
    centroid2 = np.mean(dist2_data, axis=0)
    
    # compute a pooled covariance matrix
    combined_data = np.vstack([dist1_data, dist2_data])
    covariance_matrix = np.cov(combined_data, rowvar=False)
    
    # pseudo-inverse of the covariance matrix in case it is singular
    inv_covariance = np.linalg.pinv(covariance_matrix)
    
    distance = mahalanobis(centroid1, centroid2, inv_covariance)
    
    return distance


def calc_emd(dist1_data, dist2_data): # wasserstein metric-1
    X = dist1_data
    Y = dist2_data

    # assuming equal probability for each sample
    a = np.ones(X.shape[0]) / X.shape[0]
    b = np.ones(Y.shape[0]) / Y.shape[0]

    emd_distance = wasserstein_distance_nd(X, Y, a, b)
    
    return emd_distance


def calc_emd_pot(X: np.ndarray, Y: np.ndarray): # optimised way
    import ot  
    # X: (n, d), Y: (m, d)
    n, m = X.shape[0], Y.shape[0]
    a = np.ones(n) / n
    b = np.ones(m) / m

    # build cost matrix in C
    M = ot.dist(X, Y, metric='euclidean')  

    # exact EMD (squared L2 cost); emd2 returns the *value* of the LP min ⟨M, P⟩
    return ot.emd2(a, b, M)


def calc_kl_divergence(dist1_data, dist2_data): # discrete KL divergence
    X = dist1_data
    Y = dist2_data
    
    # average representations
    mean0 = np.mean(X, axis=0)
    mean1 = np.mean(Y, axis=0)
    
    prob0 = softmax(mean0)
    prob1 = softmax(mean1)
    
    kl_divergence = entropy(prob0, prob1)
    
    return kl_divergence


def calc_gaussian_kl_divergence(dist1_data, dist2_data): # Gaussian KL divergence
    X = dist1_data
    Y = dist2_data

    # mean vectors
    mu0 = np.mean(X, axis=0)
    mu1 = np.mean(Y, axis=0)

    # variances + small constant for numerical stability
    var0 = np.var(X, axis=0) + 1e-6
    var1 = np.var(Y, axis=0) + 1e-6

    # for the diagonal covariance, the log-determinant is the sum of the log variances
    log_det_cov0 = np.sum(np.log(var0))
    log_det_cov1 = np.sum(np.log(var1))

    # trace term (since Σ₁⁻¹ is simply 1/var1 for each diagonal element)
    trace_term = np.sum(var0 / var1)

    # quadratic term:
    quad_term = np.sum(((mu1 - mu0) ** 2) / var1)

    # dimensionality of the feature space
    d = mu0.shape[0]  

    kl_divergence = 0.5 * (log_det_cov1 - log_det_cov0 - d + trace_term + quad_term)
    
    return kl_divergence


def calc_js_divergence(dist1_data, dist2_data): # Jensen-Shannon divergence, TODO: think if useful
    raise NotImplementedError("JS divergence not implemented yet")


MEASURE_LOOKUP = {
    "euclidean": calc_euclidean_distance,
    "cosine": calc_cosine_similarity,
    "mahalanobis": calc_mahalanobis_distance,
    # "emd": calc_emd,
    "emd": calc_emd_pot,
    "kl_div": calc_kl_divergence,
    "gaussian_kl_div": calc_gaussian_kl_divergence
}

class SimilarityMeasure():
    
    def __init__(self, measure="emd", similarity_transform="none", gamma=1.0):
        try:
            self.measure_func = MEASURE_LOOKUP[measure]
        except KeyError:
            raise ValueError(f"Invalid measure: {measure}")
        
        if similarity_transform not in {"none", "normalised", "exponential"}:
            raise ValueError("similarity_transform must be 'none', 'normalised', or 'exponential'")
        
        self.similarity_transform = similarity_transform
        self.gamma = gamma
    
    
    def measure(self, dist1_data, dist2_data):
        dist = self.measure_func(dist1_data, dist2_data)

        if self.similarity_transform == "normalised":
            raise ValueError("Normalised similarity transform is only supported for distance matrices")

        elif self.similarity_transform == "exponential":
            sim = np.exp(-self.gamma * dist)
            return sim

        return dist

    
    def distance_matrix(self, data):
        if data.ndim != 3:
            raise ValueError("Data must be in the form (C, N, F), C # of classes, N # of samples, F # of feature dimensions")
        
        n_classes = data.shape[0] 
        matrix = np.zeros((n_classes, n_classes))

        for i in range(n_classes):
            for j in range(i, n_classes):
                X, Y = data[i], data[j]
                dist = self.measure_func(X, Y)
                matrix[i, j] = matrix[j, i] = dist  # symmetry

        if self.similarity_transform == "normalised":
            min_val = np.min(matrix)
            max_val = np.max(matrix)
            if max_val - min_val < 1e-8:
                matrix = np.ones_like(matrix)  # all distances are equal
            else:
                matrix = 1 - (matrix - min_val) / (max_val - min_val)

        elif self.similarity_transform == "exponential":
            matrix = np.exp(-self.gamma * matrix)
        
        return matrix
    
    
    def info(self):
        measures = list(MEASURE_LOOKUP.keys())
        print("Available similarity measures:")
        for key in measures:
            print(f"- {key}")
        return measures
