import argparse
import json
import os
import tempfile
import zipfile

import requests
import tensorflow as tf


class ZippedTIOBundleMisspecificationError(Exception):
    """
    Raised in the process of a zipped tiobundle build if one or more of:
    1. model.json
    2. tflite binary
    does not exist as a file.
    """
    pass


class ZippedTIOBundleExistsError(Exception):
    """
    Raised in the process of a zipped tiobundle build if a file (or directory) already exists
    at the specified build path.
    """
    pass


def write_assets_to_zipfile(assets_dir, zfile, zip_subdir):
    """
    Recursively writes the contents of assets directory into assets/ directory in zipfile.

    Raises a TIOZipError if there is an issue writing the assets from assets_dir into the zipfile
    at the given zip_subdir.

    Args:
    1. assets_dir - Local or GCS path to be written into zfile
    2. zfile - zipfile.ZipFile instance representing the zipfile into which assets should be
       written
    3. zip_subdir - Path in zipfile under which to write the assets at the given assets_dir

    Returns: None
    """
    assets = tf.gfile.Glob(os.path.join(assets_dir, '*'))
    # Will map asset subdirectories to their target zip subdirectories
    assets_subdirs = {}
    for asset in assets:
        asset_basename = os.path.basename(asset)
        if tf.gfile.IsDirectory(asset):
            # The zip subdirectory into which the asset subdirectory should be written is formed
            # by joining the current zip_subdir with the asset_basename
            zip_target = os.path.join(zip_subdir, asset_basename)
            assets_subdirs[asset] = zip_target
        else:
            zip_target = os.path.join(zip_subdir, asset_basename)
            try:
                with tf.gfile.Open(asset, 'rb') as asset_file:
                    asset_bytes = asset_file.read()
                zfile.writestr(zip_target, asset_bytes)
            except Exception as err:
                message = 'Error inserting {} into zipfile at {}: {}'.format(asset, zip_target, err)
                raise TIOZipError(message)

    for assets_subdir in assets_subdirs:
        write_assets_to_zipfile(assets_subdir, zfile, assets_subdirs[assets_subdir])

    return None


def tiobundle_build(model_path, model_json_path, assets_path, bundle_name, outfile):
    """
    Builds zipped tiobundle file (e.g. for direct download into Net Runner)

    Args:
    1. model_path - Path to TFLite binary or SavedModel directory
    2. model_json_path - Path to TensorIO-compatible model.json file
    3. assets_path - Path to TensorIO-compatible assets directory
    4. bundle_name - Name of the bundle
    5. outfile - Name under which the zipped tiobundle file should be stored

    Returns: outfile path if the zipped tiobundle was created successfully
    """
    if tf.gfile.Exists(outfile):
        raise ZippedTIOBundleExistsError(
            'ERROR: Specified zipped tiobundle output path ({}) already exists'.format(outfile)
        )

    if not tf.gfile.Exists(model_path):
        raise ZippedTIOBundleMisspecificationError(
            'ERROR: TFLite binary path ({}) does not exist'.format(
                model_path
            )
        )

    if not tf.gfile.Exists(model_json_path) or tf.gfile.IsDirectory(model_json_path):
        raise ZippedTIOBundleMisspecificationError(
            'ERROR: model.json path ({}) either does not exist or is not a file'.format(
                model_path
            )
        )

    _, temp_outfile = tempfile.mkstemp(suffix='.zip')
    with zipfile.ZipFile(temp_outfile, 'w') as tiobundle_zip:
        # We have to use the ZipFile writestr method because there is no guarantee that
        # all the files to be included in the archive are on the same filesystem that
        # the function is running on -- they could be on GCS.
        with tf.gfile.Open(model_json_path, 'rb') as model_json_file:
            model_json = model_json_file.read()
            model_json_string = model_json.decode('utf-8')
            bundle_spec = json.loads(model_json_string)
        tiobundle_zip.writestr(
            os.path.join(bundle_name, 'model.json'),
            model_json
        )

        model_spec = bundle_spec.get('model', {})
        if tf.gfile.IsDirectory(model_path):
            # We are bundling a SavedModel directory.
            # It goes into the train/ subdirectory of bundle
            model_dirname = model_spec.get('file')
            if model_dirname is None:
                raise InvalidBundleSpecification('No "file" specified under "model" key')
            saved_model_target = os.path.join(bundle_name, model_dirname)
            write_assets_to_zipfile(model_path, tiobundle_zip, saved_model_target)
        else:
            # We are bundling a tflite file.
            # We will store the tflite file under the model_filename specified in the model.json
            # If this is not specified, we store the file as "model.tflite"
            model_filename = model_spec.get('file', 'model.tflite')
            with tf.gfile.Open(model_path, 'rb') as tflite_file:
                tflite_model = tflite_file.read()
            tiobundle_zip.writestr(
                os.path.join(bundle_name, model_filename),
                tflite_model
            )

        if assets_path is not None:
            assets_zip_target = os.path.join(bundle_name, 'assets')
            write_assets_to_zipfile(assets_path, tiobundle_zip, assets_zip_target)

    tf.gfile.Copy(temp_outfile, outfile)
    os.remove(temp_outfile)

    return outfile

def generate_argument_parser():
    """
    Generates an argument parser for use with the TensorIO Bundler CLI; also used by bundlebot

    Args: None

    Returns: None
    """
    parser = argparse.ArgumentParser(description='Create tiobundles for use with TensorIO')

    parser.add_argument(
        '--tflite-model',
        required=False,
        help='Path to TFLite model (GCS allowed)'
    )
    parser.add_argument(
        '--model-json',
        required=True,
        help='Path to TensorIO model.json file'
    )
    parser.add_argument(
        '--assets-dir',
        required=False,
        help='Path to assets directory'
    )
    parser.add_argument( '--bundle-name',
        required=True,
        help='Name of tiobundle'
    )
    parser.add_argument(
        '--outfile',
        required=False,
        help='Path at which tiobundle zipfile should be created; defaults to <BUNDLE_NAME>.zip'
    )

    return parser


if __name__ == '__main__':
    parser = generate_argument_parser()
    args = parser.parse_args()
    tiobundle_build(args.tflite_model, args.model_json, args.assets_dir, args.bundle_name, args.outfile)