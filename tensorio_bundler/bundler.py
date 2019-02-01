import argparse
import json
import os
import tempfile
import zipfile

import tensorflow as tf

class SavedModelDirMisspecificationError(Exception):
    """
    Raised in the process of a TFLite build if the SavedModel directory either does not
    exist or is not a directory.
    """
    pass

class TFLiteFileExistsError(Exception):
    """
    Raised in the process of a TFLite build if a file (or directory) already exists
    at the specified build path.
    """
    pass

class ZippedTFBundleExistsError(Exception):
    """
    Raised in the process of a zipped tfbundle build if a file (or directory) already exists
    at the specified build path.
    """
    pass

class ZippedTFBundleMisspecificationError(Exception):
    """
    Raised in the process of a zipped tfbundle build if one or more of:
    1. model.json
    2. tflite binary
    does not exist as a file.
    """
    pass

def tflite_build_from_saved_model(saved_model_dir, outfile):
    if tf.gfile.Exists(outfile):
        raise TFLiteFileExistsError(
            'ERROR: Specified TFLite binary path ({}) already exists'.format(outfile)
        )
    if not tf.gfile.Exists(saved_model_dir) or not tf.gfile.IsDirectory(saved_model_dir):
        raise SavedModelDirMisspecificationError(
            ('ERROR: Specified SavedModel directory ({}) either does not exist or is not a '
             'directory').format(saved_model_dir)
        )

    converter = tf.contrib.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()
    with tf.gfile.Open(outfile, 'wb') as outf:
        outf.write(tflite_model)

def tfbundle_build(tflite_path, model_json_path, assets_path, bundle_name, outfile):
    if tf.gfile.Exists(outfile):
        raise ZippedTFBundleExistsError(
            'ERROR: Specified zipped tfbundle output path ({}) already exists'.format(outfile)
        )

    if not tf.gfile.Exists(tflite_path) or tf.gfile.IsDirectory(tflite_path):
        raise ZippedTFBundleMisspecificationError(
            'ERROR: TFLite binary path ({}) either does not exist or is not a file'.format(
                tflite_path
            )
        )

    if not tf.gfile.Exists(model_json_path) or tf.gfile.IsDirectory(model_json_path):
        raise ZippedTFBundleMisspecificationError(
            'ERROR: model.json path ({}) either does not exist or is not a file'.format(
                tflite_path
            )
        )

    _, temp_outfile = tempfile.mkstemp(suffix='.zip')
    with zipfile.ZipFile(temp_outfile, 'w') as tfbundle_zip:
        # We have to use the ZipFile writestr method because there is no guarantee that
        # all the files to be included in the archive are on the same filesystem that
        # the function is running on -- they could be on GCS.
        with tf.gfile.Open(model_json_path, 'rb') as model_json_file:
            model_json = model_json_file.read()
            model_json_string = model_json.decode('utf-8')
            model_spec = json.loads(model_json_string)
        # We will store the tflite file under the model_filename specified in the model.json
        # If this is not specified, we store the file as "model.tflite"
        tflite_spec = model_spec.get('model', {})
        model_filename = tflite_spec.get('file', 'model.tflite')
        tfbundle_zip.writestr(
            os.path.join(bundle_name, 'model.json'),
            model_json
        )

        with tf.gfile.Open(tflite_path, 'rb') as tflite_file:
            tflite_model = tflite_file.read()
        tfbundle_zip.writestr(
            os.path.join(bundle_name, model_filename),
            tflite_model
        )

        if assets_path is not None:
            # TODO(nkashy1): Generalize this so that the assets are written in recursively
            # As it stands, the assumption is that the assets directory is flat and contains
            # no subdirectories
            assets = tf.gfile.Glob(os.path.join(assets_path, '*'))
            for asset in assets:
                asset_basename = os.path.basename(asset)
                with tf.gfile.Open(asset, 'rb') as asset_file:
                    asset_bytes = asset_file.read()
                tfbundle_zip.writestr(
                    os.path.join(bundle_name, 'assets', asset_basename),
                    asset_bytes
                )

    tf.gfile.Copy(temp_outfile, outfile)
    os.remove(temp_outfile)

    return outfile

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create tfbundles for use with TensorIO')

    parser.add_argument(
        '--build',
        action='store_true',
        help='Specifies whether a TFLite file should be built at the specified tflite model path'
    )
    parser.add_argument(
        '--saved-model-dir',
        required=False,
        help='Path to SavedModel pb file and variables'
    )
    parser.add_argument(
        '--tflite-model',
        required=True,
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
        help='(Optional) Path to assets directory'
    )
    parser.add_argument(
        '--bundle-name',
        required=True,
        help='Name of tfbundle'
    )
    parser.add_argument(
        '--outfile',
        required=False,
        help='(Optional) Path at which tfbundle zipfile should be created'
    )

    args = parser.parse_args()
    if args.build:
        if args.saved_model_dir is None:
            raise ValueError(
                'ERROR: When calling script with --build enabled, specify --saved-model-dir'
            )
        if tf.gfile.Exists(args.tflite_model):
            raise Exception('ERROR: TFLite model already exists - {}'.format(args.tflite_model))

        print('Building TFLite model -')
        print('SavedModel directory: {}, TFLite model: {}'.format(
            args.saved_model_dir, args.tflite_model
        ))
        tflite_build_from_saved_model(args.saved_model_dir, args.tflite_model)

    tfbundle_zip = args.outfile
    if tfbundle_zip is None:
        tfbundle_zip = '{}.zip'.format(args.bundle_name)

    print('Building tfbundle -')
    print('TFLite model: {}, model.json: {}, assets directory: {}, bundle: {}, zipfile: {}'.format(
        args.tflite_model,
        args.model_json,
        args.assets_dir,
        args.bundle_name,
        tfbundle_zip
    ))
    tfbundle_build(args.tflite_model, args.model_json, args.assets_dir, args.bundle_name, tfbundle_zip)

    print('Done!')
