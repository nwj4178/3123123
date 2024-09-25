import numpy as np

def softmax(logits):
    exp_values = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Overflow 방지
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities

class CrossEntropy:
    def forward(self, predictions, targets):

        predictions = softmax(predictions)

        predictions = np.clip(predictions, 1e-7, 1 - 1e-7)

        if targets.ndim == 1:
            correct_confidences = predictions[np.arange(len(predictions)), targets]
        else:
            correct_confidences = np.sum(predictions * targets, axis=1)

        negative_log_likelihoods = -np.log(correct_confidences)

        return np.mean(negative_log_likelihoods)

logits = np.array([
    [2.0, 1.0, 0.1],  # 첫 번째 데이터의 로짓 값
    [1.0, 3.0, 0.2],  # 두 번째 데이터의 로짓 값
    [0.5, 0.7, 2.5]   # 세 번째 데이터의 로짓 값
])

targets = np.array([0, 1, 2])

ce = CrossEntropy()

loss = ce.forward(logits, targets)
print("Categorical Cross-Entropy Loss:", loss)