import pytest
from ml.model import compute_model_metrics



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


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    """
    # add description for the second test
    """
    # Your code here
    pass


# TODO: implement the third test. Change the function name and input as needed
def test_three():
    """
    # add description for the third test
    """
    # Your code here
    pass
