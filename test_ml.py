import pytest
from ml.model import compute_model_metrics, inference, train_model



def test_compute_model_metrics():
    """
    # Test that compute_model_metrics returns the expected metrics 
    """
    y= [1, 1, 0, 0]
    preds = [1, 0, 0, 0]

    precision, recall, fbeta = compute_model_metrics(y,preds)

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(0.5)
    assert fbeta == pytest.approx(0.6667, abs=0.001)

class FakeModel:
    def predict(self, X):
        return[0, 1, 1]

def test_inference():
    """
    # Tests to make sure that inference returns the models predictions 
    """
    X = [[1], [2], [3]]
    model = FakeModel()

    predictions = inference(model, X)

    assert list(predictions) == [0, 1, 1]

X_train = [
    [1, 10],
    [2, 20],
    [3, 30],
    [4, 40]
]

y_train = [0, 0, 1, 1]

def test_train_model():
    """
    # Test that train_model creates a trained model 
    """
    X_train = [
        [1, 10],
        [2, 20],
        [3, 30],
        [4, 40]
    ]

    y_train = [0, 0, 1, 1]

    model = train_model(X_train, y_train)
   
    predictions = model.predict(X_train)

    assert len(predictions) == len(y_train)