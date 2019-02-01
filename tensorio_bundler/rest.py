import json

import falcon

from . import bundler

class PingHandler:
    """
    Handler for uptime checks
    """
    def on_get(self, req, resp):
        """
        Returns status code 200 with body "ok" on GET requests. Intended for uptime
        checks.
        """
        resp.status = falcon.HTTP_200
        resp.body = 'ok'

    # def on_post(self, req, resp):
    #     resp.status = falcon.HTTP_200
    #     resp.body = json.dumps(req.media['lol'])

class BundleHandler:
    """
    Handler for bundle creation requests
    """
    def on_post(self, req, resp):
        """
        Accepts POST requests to create a tfbundle from:
        1. A model.json file (GCS path)
        2. An assets directory (GCS path)
        3. A TFLite binary path and a SavedModel binary path (in the case that a SavedModel binary
           is specified, the caller must also specify explicity that a TFLite build step is
           required); the path to the TFLite file should be passed anyway, and if the handler is
           required to build the TFLite binary from the SavedModel binary, then it builds it at that
           path. (GCS paths)
        4. Bundle name
        5. Bundle output path

        Possible responses:
        + Responds with status code 200 and body containing the GCS path of the tfbundle if the
          bundle was created successfully.
        + Responds with a status code of 400 if the request body is either not a processable JSON
          string or if it does not specify the appropriate fields or if the fields are inappropriate
          to the request (e.g. missing keys). The body of the response will specify the erroneous
          conditions.
        + Responds with a status code of 409 if a file already exists at the given tfbundle path.
          The response body will be a string specifying the GCS path and stating that a file already
          exists there.
        + Responds with a status code of 422 if a SavedModel path is specified but the build flag is
          not set in the JSON body of the request.
        + Responds with a status code of 409 if a SavedModel path is specified and the build flag is
          set to true, but if there is already a file at the specified TFLite path.
        + Responds with a 404 if one or more of the following assets is not found:
            + model.json
            + assets directory
            + TFlite binary if SavedModel binary not specified, SavedModel binary otherwise
            + "build": true, if SavedModel binary is specified
        """
        # The following assignment automatically returns a 400 response code if the input is not
        # parseable JSON.
        request_body = req.media
        expected_keys = {
            'model_json',
            'tflite_binary',
            'bundle_name',
            'bundle_output_path'
        }
        missing_keys = [key for key in expected_keys if key not in request_body]
        if len(missing_keys) > 0:
            message = 'Request body missing the following keys: {}'.format(
                ', '.join(missing_keys)
            )
            raise falcon.HTTPBadRequest(message)

        resp.status = falcon.HTTP_200
        resp.body = 'ok'

api = falcon.API()

ping = PingHandler()
api.add_route('/ping', ping)

bundler = BundleHandler()
api.add_route('/bundle', bundler)
