import pytest
from ml.model import compute_model_metrics, inference



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


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
