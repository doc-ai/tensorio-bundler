"""
bundlebot - the TensorIO Bundler Slack bot
"""

import os

from slack_invoker import invoke, parse
import requests

from . import bundler

def generate_bundler_client(bundler_rest_api_url):
    """
    Generates a client for a given TensorIO Bundler REST API instance.

    Args:
    1. bundler_rest_api_url - URL (<host>:<port>) for a TensorIO Bundler Rest API

    Returns: Function which can be called to make REST requests against the given REST API
    """
    def bundler_client(
        build,
        saved_model_dir,
        tflite_model,
        model_json,
        assets_dir,
        bundle_name,
        outfile):
        """
        Functional client to TensorIO Bundler REST API

        Args:
        1. build - Specifies whether or not to build from SavedModel dir
        2. saved_model_dir - Directory containing SavedModel protobuf and variables
        3. tflite_model - Path to TFLite binary
        4. model_json - Path to TensorIO-compatible model.json file
        5. assets_dir - Path to TensorIO-compatible assets directory
        6. bundle_name - Name of TensorIO tfbundle
        7. outfile - Path to which to write zipped tfbundle file

        Returns: Path at which zipped tfbundle was stored
        """
        payload = {
            'build': build,
            'saved_model_dir': saved_model_dir,
            'tflite_path': tflite_model,
            'model_json_path': model_json,
            'assets_path': assets_dir,
            'bundle_name': bundle_name,
            'bundle_output_path': outfile
        }

        response = requests.post(bundler_rest_api_url, json=payload)
        return response.text

    return bundler_client

if __name__ == '__main__':
    bot_user_access_token = os.environ.get('SLACK_BOT_USER_OAUTH_ACCESS_TOKEN')
    if bot_user_access_token is None:
        raise ValueError('ERROR: SLACK_BOT_USER_OAUTH_ACCESS_TOKEN environment variable not set')

    rtm_poll_interval = int(os.environ.get('RTM_POLL_INTERVAL', '1'))

    bundler_url = os.environ.get('BUNDLER_URL')
    if bundler_url is None:
        raise ValueError('ERROR: BUNDLER_URL environment variable not set')

    bundler_client = generate_bundler_client(bundler_url)

    parser = bundler.generate_argument_parser()
    bundlebot = parse.wrap_runner(parser, bundler_client)
    invoke.rtm_bot_user(bundlebot, bot_user_access_token, rtm_poll_interval)
