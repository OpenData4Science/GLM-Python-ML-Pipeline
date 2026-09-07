import unittest
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient
from sklearn.exceptions import InconsistentVersionWarning
import app


SAMPLE = {
    'age': 45, 'tenure': 24, 'monthly_charges': 79.85,
    'total_charges': 1800, 'contract_type': 'Month-to-month',
    'payment_method': 'Electronic check',
}


class ApiTest(unittest.TestCase):
    def test_missing_artifacts_report_unavailable(self):
        with patch.object(app, 'model', None), patch.object(app, 'scaler', None):
            with TestClient(app.app) as client:
                self.assertEqual(client.get('/health').status_code, 503)
                self.assertEqual(client.post('/predict', json=SAMPLE).status_code, 503)

    def test_loader_rejects_unknown_feature_order(self):
        with patch.object(app.joblib, 'load', side_effect=[SimpleNamespace(classes_=[0, 1]), SimpleNamespace(n_features_in_=3)]):
            self.assertEqual(app.load_artifacts(), (None, None))

    def test_loader_rejects_version_mismatch(self):
        warning = InconsistentVersionWarning(estimator_name='LogisticRegression', current_sklearn_version='1.6.1', original_sklearn_version='1.1.3')
        with patch.object(app.joblib, 'load', side_effect=warning):
            self.assertEqual(app.load_artifacts(), (None, None))

    def test_prediction_failure_does_not_disclose_internal_details(self):
        scaler = MagicMock()
        scaler.transform.side_effect = RuntimeError('private-provider-or-filesystem-detail')
        with patch.object(app, 'model', MagicMock()), patch.object(app, 'scaler', scaler):
            with TestClient(app.app) as client:
                result = client.post('/predict', json=SAMPLE)
                self.assertEqual(result.status_code, 503)
                self.assertNotIn('private-provider', result.text)

    def test_valid_prediction_flow_with_offline_doubles(self):
        import numpy as np
        model, scaler = MagicMock(), MagicMock()
        model.predict_proba.return_value = np.array([[0.75, 0.25]])
        with patch.object(app, 'model', model), patch.object(app, 'scaler', scaler), patch.object(app, 'summarise_prediction', return_value='Offline test summary'):
            with TestClient(app.app) as client:
                self.assertEqual(client.get('/health').status_code, 200)
                result = client.post('/predict', json=SAMPLE)
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.json()['churn_probability'], 0.25)


if __name__ == '__main__':
    unittest.main()
