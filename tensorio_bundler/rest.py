"""
TensorIO Bundler REST API
"""

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

class BundleHandler:
    """
    Handler for bundle creation requests
    """

    required_keys = {
        'tflite_path',
        'model_json_path',
        'assets_path',
        'bundle_name',
        'bundle_output_path'
    }

    def on_post(self, req, resp):
        """
        Accepts POST requests to create a tiobundle from:
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
        + Responds with status code 200 and body containing the GCS path of the tiobundle if the
          bundle was created successfully.
        + Responds with a status code of 400 if the request body is either not a processable JSON
          string or if it does not specify the appropriate fields or if the fields are inappropriate
          to the request (e.g. missing keys). The body of the response will specify the erroneous
          conditions.
        + Responds with a status code of 409 if a file already exists at the given tiobundle path.
          The response body will be a string specifying the GCS path and stating that a file already
          exists there.
        + Responds with a status code of 422 if a SavedModel path is specified but the build flag is
          not set in the JSON body of the request.
        + Responds with a status code of 422 if the build flag is set to true but a SavedModel path
          is not specified in the JSON body of the request.
        + Responds with a status code of 409 if a SavedModel path is specified and the build flag is
          set to true, but if there is already a file at the specified TFLite path.
        + Responds with a 404 if one or more of the following is not found:
            + model.json
            + assets directory
            + TFlite binary if SavedModel binary not specified, SavedModel binary otherwise
        """
        # The following assignment automatically returns a 400 response code if the input is not
        # parseable JSON.
        request_body = req.media

        missing_keys = [key for key in self.required_keys if key not in request_body]
        if len(missing_keys) > 0:
            message = 'Request body missing the following keys: {}'.format(
                ', '.join(missing_keys)
            )
            raise falcon.HTTPBadRequest(message)

        if request_body.get('saved_model_dir') is not None and request_body.get('build') != True:
            raise falcon.HTTPUnprocessableEntity(
                description=(
                    'ERROR: "build" must be set to true in request body if "saved_model_dir" is'
                    'specified'
                )
            )

        if request_body.get('build') is True and request_body.get('saved_model_dir') is None:
            raise falcon.HTTPUnprocessableEntity(
                description=(
                    'ERROR: "saved_model_dir" must be specified in request body if "build" is'
                    'set to true'
                )
            )

        if request_body.get('build') is True:
            try:
                bundler.tflite_build_from_saved_model(
                    request_body.get('saved_model_dir'),
                    request_body.get('tflite_path')
                )
            except bundler.TFLiteFileExistsError as e:
                raise falcon.HTTPConflict(description=str(e))
            except bundler.SavedModelDirMisspecificationError as e:
                raise falcon.HTTPNotFound(description=str(e))
            except Exception as e:
                raise falcon.HTTPInternalServerError()

        try:
            outfile = bundler.tiobundle_build(
                request_body.get('tflite_path'),
                request_body.get('model_json_path'),
                request_body.get('assets_path'),
                request_body.get('bundle_name'),
                request_body.get('bundle_output_path')
            )
        except bundler.ZippedTIOBundleExistsError as e:
            raise falcon.HTTPConflict(description=str(e))
        except bundler.ZippedTIOBundleMisspecificationError as e:
            raise falcon.HTTPNotFound(description=str(e))
        except Exception:
            raise falcon.HTTPInternalServerError()

        resp.status = falcon.HTTP_200
        resp.body = outfile

api = falcon.API()

ping_handler = PingHandler()
api.add_route('/ping', ping_handler)

bundle_handler = BundleHandler()
api.add_route('/bundle', bundle_handler)
