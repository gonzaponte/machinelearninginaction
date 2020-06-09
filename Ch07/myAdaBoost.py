import numpy as np
from utils import npmap, load_data

def load_data_set():
    dataset = np.array([[1. , 2.1],
                       [2. , 1.1],
                       [1.3, 1. ],
                       [1. , 1. ],
                       [2. , 1. ]])
    labels  = np.array([1.0, 1.0, -1.0, -1.0, 1.0])
    return dataset, labels


def load_horse_colic_data_set(filename="horseColicTraining2.txt"):
    return load_data(filename)


def classify_stump(data, threshold, comparator):
    return 1 - 2*comparator(data, threshold)


def stump(data, labels, weights, nsteps=11):
    min_error = np.inf
    for ifeature, feature in enumerate(data.T):
        thresholds = np.linspace(np.min(feature), np.max(feature), nsteps)
        for threshold in thresholds:
            for comparator in [np.less_equal, np.greater]:
                prediction = classify_stump(feature, threshold, comparator)
                error = weights.dot(prediction != labels)
                if error < min_error:
                    min_error = error
                    best_prediction = prediction
                    best_feature    = ifeature
                    best_threshold  = threshold
                    best_comparator = comparator
    return {"prediction": best_prediction,
            "feature"   : best_feature,
            "threshold" : best_threshold,
            "comparator": best_comparator,
            "error"     :  min_error}


def train_adaboost(data, labels, max_error=1e-6, nsteps=11, max_iterations=100000):
    outputs = []
    weights = np.ones_like(labels)/labels.size
    prediction = np.zeros_like(labels)
    error_rate = max_error + 1
    for i in range(max_iterations):
        output = stump(data, labels, weights, nsteps)
        error_rate = max(output["error"], 1e-16)
        alpha = 0.5 * np.log(1/error_rate - 1)
        output["alpha"] = alpha
        outputs.append(output)
        weights *= np.exp(-alpha * labels * output["prediction"])
        weights /= np.sum(weights)
        prediction += alpha * output["prediction"]
        error_rate = np.count_nonzero(np.sign(prediction) != labels)/labels.size
        if error_rate <= max_error:
            break
    return outputs


def get_classifier(outputs):
    def classify_adaboost(data):
        if len(np.shape(data)) == 1:
            data = data[np.newaxis]
        data = data.T
        prediction = np.zeros(data.shape[1])
        for output in outputs:
            prediction += classify_stump(data[output["feature"]],
                                         output["threshold"],
                                         output["comparator"]) * output["alpha"]
        return np.sign(prediction)
    return classify_adaboost


def test_horse_colic(**kwargs):
    hc_dataset     , hc_labels      = load_horse_colic_data_set('horseColicTraining2.txt')
    hc_dataset_test, hc_labels_test = load_horse_colic_data_set('horseColicTest2.txt')
    
    hc_classifier_params = train_adaboost(hc_dataset, hc_labels, **kwargs)
    hc_classifier        = get_classifier(hc_classifier_params)
    
    hc_prediction = hc_classifier(hc_dataset_test)
    error_rate    = 100 * np.count_nonzero(hc_prediction != hc_labels_test)/hc_labels_test.size
    print(error_rate)
    return error_rate