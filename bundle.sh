#!/usr/bin/env bash

# Arguments:
# 1. MODEL_DIR - path to model directory, example: gs://doc-ai-models/...
# 2. OUTPUT_NODES - names of output nodes in TensorFlow graph, comma separated, example: custom_layers/Softmax
# 3. INPUT_NODES - names of input nodes in TensorFlow graph, coma-separated, example: serving-input-placeholder
# 4. INPUT_SHAPES - shapes of input tensors, comma-separated within each tensor, colon-separated across tensors, example: 1,224,224,3
# 5. OUTPUT_FILE - file to which to write toco-optimized graph, example: model.tflite

set -x

MODEL_DIR=$1
OUTPUT_NODES=$2
INPUT_NODES=$3
INPUT_SHAPES=$4
OUTPUT_FILE=$5

OUTPUT_GRAPH=$(mktemp --suffix ".pb")
echo "Writing frozen graph to: $OUTPUT_GRAPH ..."

freeze_graph \
    --input_saved_model_dir $MODEL_DIR \
    --output_graph $OUTPUT_GRAPH \
    --output_node_names $OUTPUT_NODES


echo "Running toco..."
toco \
    --graph_def_file $OUTPUT_GRAPH \
    --output_file $OUTPUT_FILE \
    --inference_type FLOAT \
    --input_arrays $INPUT_NODES \
    --input_shapes $INPUT_SHAPES \
    --output_arrays $OUTPUT_NODES

echo "Done!"
