#!/usr/bin/env bash

# Produces a tfbundle file from:
# 1. A TensorFlow SavedModel binary
# 2. A model.json file
# 3. An assets directory
#
# Assumes:
# 1. Access to a Python with TensorFlow installed (makes use of freeze_graph and toco binaries
#    in addition to some Python scripts which import tensorflow).
#
# For more information, run this script with the "-h" flag (or read the usage output below)

set -e -o pipefail

USAGE="Usage: $0 MODEL_DIR INPUT_NODES INPUT_SHAPES OUTPUT_NODES OUTPUT_FILE"
function usage() {
cat <<EOF
$USAGE

ARGUMENTS
    MODEL_DIR
        Directory containing TensorFlow SavedModel binary and variables. (GCS allowed)
        Example: gs://doc-ai-models/happy-face/batch-4-ing-rate-5e-7-true/export/LatestExporter/1548717255/

    INPUT_NODES
        Names of input nodes in TensorFlow graph. Should be comma separated.
        Example: serving-input-placeholder

    INPUT_SHAPES
        Shapes of the tensors corresponding to the INPUT_NODES. Shapes are comma-separated
        representations of shape tuples. Shapes for different tensors should be separated
        with colons (:).
        Example: 1,224,224,3

    OUTPUT_NODES
        Comma-separated list of names of output nodes in the TensorFlow graph.
        Example: custom_layers/Softmax

    OUTPUT_FILE
        Path to which to write toco-optimized graph. (GCS allowed)
        Example: model.tflite

ENVIRONMENT VARIABLES
This script also respects the following environment variables:
    DEBUG
        If set to "true", the script will print the temporary files and directories it
        creates to stdout, as well as information about its progress.

    GOOGLE_APPLICATION_CREDENTIALS
        Used by TensorFlow utilities if any of the input or output paths are Google Cloud
        Storage paths. Should point at a JSON file containing credentials for a service account
        that has the appropriate access to the specified buckets/objects.
        For more information, read the Google Cloud Platform documentation:
            https://cloud.google.com/docs/authentication/getting-started
EOF
}

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
    exit 2
fi

DEBUG=${DEBUG:-false}

MODEL_DIR=$1
INPUT_NODES=$2
INPUT_SHAPES=$3
OUTPUT_NODES=$4
OUTPUT_FILE=$5

OUTPUT_GRAPH=$(mktemp --suffix ".pb")
if [ "$DEBUG" = "true" ]; then
    echo "Writing frozen graph to: $OUTPUT_GRAPH ..."
fi

freeze_graph \
    --input_saved_model_dir $MODEL_DIR \
    --output_graph $OUTPUT_GRAPH \
    --output_node_names $OUTPUT_NODES

if [ "$DEBUG" = "true" ]; then
    echo "Running toco..."
fi
toco \
    --graph_def_file $OUTPUT_GRAPH \
    --output_file $OUTPUT_FILE \
    --inference_type FLOAT \
    --input_arrays $INPUT_NODES \
    --input_shapes $INPUT_SHAPES \
    --output_arrays $OUTPUT_NODES

if [ "$DEBUG" = "true" ]; then
    echo "Done!"
fi
