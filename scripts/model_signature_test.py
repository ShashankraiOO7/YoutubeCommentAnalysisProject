import mlflow
import pytest
import pandas as pd
import pickle
import numpy as np
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
TRACKING_URI = "http://ec2-3-110-174-70.ap-south-1.compute.amazonaws.com:5000/"
TEST_INPUTS = [
    ("hi how are you", True),  # (text, expected_to_pass)
    ("test query", True),
    ("", False),  # Empty string might fail
    (" "*100, False),  # Whitespace might fail
    (123, False),  # Invalid type
    (None, False)  # None input
]

@pytest.mark.parametrize("model_name, stage, vectorizer_path", [
    ("yt_chrome_plugin_model", "staging", "tfidf_vectorizer.pkl"),
])
def test_model_with_vectorizer(model_name, stage, vectorizer_path):
    """Test model with external vectorizer"""
    # Setup MLflow client
    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    
    try:
        # Verify model exists
        latest_version_info = client.get_latest_versions(model_name, stages=[stage])
        assert latest_version_info, f"No model found in '{stage}' stage for '{model_name}'"
        latest_version = latest_version_info[0].version
        
        # Load model and vectorizer
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Loaded model {model_name} version {latest_version}")
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logger.info(f"Loaded vectorizer from {vectorizer_path}")

        # Test with various inputs
        for i, (input_text, should_pass) in enumerate(TEST_INPUTS):
            try:
                # Transform input
                if isinstance(input_text, str):
                    input_data = vectorizer.transform([input_text])
                    input_df = pd.DataFrame(
                        input_data.toarray(), 
                        columns=vectorizer.get_feature_names_out()
                    )
                    
                    # Verify input shape
                    assert input_df.shape[1] == len(vectorizer.get_feature_names_out()), \
                        "Feature dimension mismatch"
                    
                    # Predict
                    prediction = model.predict(input_df)
                    
                    # Verify output
                    assert len(prediction) == 1, "Output length mismatch"
                    assert prediction[0] is not None, "Null prediction"
                    
                    if should_pass:
                        logger.info(f"Test {i} passed - Input: '{input_text[:20]}...' -> Output: {prediction[0]}")
                    else:
                        pytest.fail(f"Test {i} should have failed but passed: '{input_text}'")
                
            except Exception as e:
                if should_pass:
                    logger.error(f"Test {i} failed unexpectedly: {str(e)}")
                    raise
                else:
                    logger.info(f"Test {i} failed as expected for input '{input_text}'")
        
        # Additional test: Verify vectorizer vocabulary matches model expectations
        if hasattr(model, 'metadata') and 'input_schema' in model.metadata.to_dict():
            input_schema = model.metadata.to_dict()['input_schema']
            if 'columns' in input_schema:
                expected_features = set(input_schema['columns'])
                actual_features = set(vectorizer.get_feature_names_out())
                assert expected_features.issuperset(actual_features), \
                    "Model expects features not in vectorizer"
                
    except MlflowException as e:
        logger.error(f"MLflow operation failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise