import json
import os
import shutil
import tempfile

from falcon import testing

from . import rest

class TestRestAPI(testing.TestCase):
    FIXTURES_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'fixtures'
    )
    TEST_MODEL_DIR = os.path.join(FIXTURES_DIR, 'test-model')
    TEST_TFLITE_FILE = os.path.join(FIXTURES_DIR, 'test.tflite')
    TEST_TIOBUNDLE = os.path.join(FIXTURES_DIR, 'test.tiobundle')

    def setUp(self):
        self.api = testing.TestClient(rest.api)
        self.temp_dirs = []
    
    def tearDown(self):
        for temp_dir in self.temp_dirs:
            shutil.rmtree(temp_dir)
    
    def create_temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.temp_dirs.append(temp_dir)
        return temp_dir

    def test_ping(self):
        result = self.api.simulate_get('/ping')
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.text, 'ok')
    
    def test_bundle_from_tflite(self):
        outdir = self.create_temp_dir()
        outfile = os.path.join(outdir, 'test.tiobundle.zip')

        body = {
            'tflite_path': os.path.join(self.TEST_TIOBUNDLE, 'model.tflite'),
            'model_json_path': os.path.join(self.TEST_TIOBUNDLE, 'model.json'),
            'assets_path': os.path.join(self.TEST_TIOBUNDLE, 'assets'),
            'bundle_name': 'actual.tiobundle',
            'bundle_output_path': outfile
        }

        result = self.api.simulate_post(
            '/bundle',
            json=body
        )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.text, body['bundle_output_path'])

    def test_bundle_from_saved_model_dir(self):
        outdir = self.create_temp_dir()
        outfile = os.path.join(outdir, 'test.tiobundle.zip')

        body = {
            'saved_model_dir': self.TEST_MODEL_DIR,
            'build': True,
            'tflite_path': os.path.join(outdir, 'model.tflite'),
            'model_json_path': os.path.join(self.TEST_TIOBUNDLE, 'model.json'),
            'assets_path': os.path.join(self.TEST_TIOBUNDLE, 'assets'),
            'bundle_name': 'actual.tiobundle',
            'bundle_output_path': outfile
        }

        result = self.api.simulate_post(
            '/bundle',
            json=body
        )

        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.text, body['bundle_output_path'])

    def test_bundle_with_malformed_request_body(self):
        result = self.api.simulate_post(
            '/bundle',
            headers={ 'Content-Type': 'application/json' },
            body='{"lol":'
        )
        self.assertEqual(result.status_code, 400)

    def test_bundle_with_empty_json_object_as_body(self):
        result = self.api.simulate_post(
            '/bundle',
            headers={ 'Content-Type': 'application/json' },
            body='{}'
        )
        self.assertEqual(result.status_code, 400)
        self.assertSetEqual(
            set(result.json.get('title', '').split(':')[-1].strip().split(', ')),
            rest.BundleHandler.required_keys
        )

    def test_bundle_with_single_missing_keys_in_body(self):
        for key in rest.BundleHandler.required_keys:
            body_dict = {
                required_key:True for required_key in rest.BundleHandler.required_keys
                if required_key != key
            }
            body = json.dumps(body_dict)
            result = self.api.simulate_post(
                '/bundle',
                headers={ 'Content-Type': 'application/json' },
                body=body
            )
            self.assertEqual(result.status_code, 400)
            self.assertSetEqual(
                set(result.json.get('title', '').split(':')[-1].strip().split(', ')),
                {key}
            )

    def test_bundle_with_saved_model_dir_but_no_build_flag(self):
        body_dict = {key:True for key in rest.BundleHandler.required_keys}
        body_dict['saved_model_dir'] = True
        body = json.dumps(body_dict)
        result = self.api.simulate_post(
            '/bundle',
            headers={ 'Content-Type': 'application/json' },
            body=body
        )
        self.assertEqual(result.status_code, 422)
    
    def test_bundle_with_saved_model_dir_but_false_build_flag(self):
        body_dict = {key:True for key in rest.BundleHandler.required_keys}
        body_dict['saved_model_dir'] = True
        body_dict['build'] = False
        body = json.dumps(body_dict)
        result = self.api.simulate_post(
            '/bundle',
            headers={ 'Content-Type': 'application/json' },
            body=body
        )
        self.assertEqual(result.status_code, 422)

    def test_bundle_with_saved_model_dir_but_string_build_flag(self):
        body_dict = {key:True for key in rest.BundleHandler.required_keys}
        body_dict['saved_model_dir'] = True
        body_dict['build'] = 'invalid'
        body = json.dumps(body_dict)
        result = self.api.simulate_post(
            '/bundle',
            headers={ 'Content-Type': 'application/json' },
            body=body
        )
        self.assertEqual(result.status_code, 422)

    def test_bundle_with_build_flag_true_and_no_saved_model_dir(self):
        body_dict = {key:True for key in rest.BundleHandler.required_keys}
        body_dict['build'] = True
        body = json.dumps(body_dict)
        result = self.api.simulate_post(
            '/bundle',
            headers={ 'Content-Type': 'application/json' },
            body=body
        )
        self.assertEqual(result.status_code, 422)
