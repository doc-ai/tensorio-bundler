from falcon import testing

from . import rest

class TestRestAPI(testing.TestCase):
    def setUp(self):
        self.api = testing.TestClient(rest.api)

    def test_ping(self):
        result = self.api.simulate_get('/ping')
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.text, 'ok')

    def test_bundle_with_malformed_request_body(self):
        result = self.api.simulate_post(
            '/bundle',
            headers={ 'Content-Type': 'application/json' },
            body='{"lol":'
        )
        self.assertEqual(result.status_code, 400)
