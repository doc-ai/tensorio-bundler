# tensorio-bundler
Create TensorIO model bundles


## Running the bundler from the command line

NOTE: Working on making a PyPI package. Once that is done, these instructions will change
to use whatever binary the corresponding `pip install` produces.

### Requirements
+ Python 3

### Instructions
The `tensorio_bundler` module comes with a `bundler` utility that you can use to create TensorIO
zipped tiobundle files directly from your command line.

For more information on how to run the `bundler`, run:
```
python -m tensorio_bundler.bundler -h
```

A sample invocation (using test data, assumed to be run from project root -- same directory as this
README):
```
python -m tensorio_bundler.bundler \
    --tflite-model ./tensorio_bundler/fixtures/test.tflite \
    --model-json ./tensorio_bundler/fixtures/test.tiobundle/model.json \
    --assets-dir ./tensorio_bundler/fixtures/test.tiobundle/assets \
    --bundle-name sample.tiobundle \
    --outfile sample.tiobundle.zip
```


## Calling the bundler locally through the REST API

To run the REST API locally from project root (same directory as this README):
```
gunicorn tensorio_bundler.rest:api
```

In a separate terminal window, you can invoke the bundler as follows:
```
TFLITE_PATH="\"$(mktemp -d)/model.tflite\""

read -r -d '' REQUEST_BODY <<-EOF
    {
        "saved_model_dir": "./tensorio_bundler/fixtures/test-model",
        "build": true,
        "tflite_path": $TFLITE_PATH,
        "model_json_path": "./tensorio_bundler/fixtures/test.tiobundle/model.json",
        "assets_path": "./tensorio_bundler/fixtures/test.tiobundle/assets",
        "bundle_name": "curl-test.tiobundle",
        "bundle_output_path": "curl-test.tiobundle.zip"
    }
EOF

curl -v -X POST \
    -H "Content-Type: application/json" \
    -d "$REQUEST_BODY" \
    http://localhost:8000/bundle
```


## Running the bundler via docker

### Requirements
+ Docker

If you don't have it, [get it](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

### Instructions
You can either bind mount the paths to the inputs into your docker container when you run the
bundler or you can bind mount in a service account credentials file and set the
`GOOGLE_APPLICATION_CREDENTIALS` environment variable to point at the mount path in the container.

NOTE: These instructions are extremely sparse at the moment. They will not be so forever.


## Running tests if you want to contribute to this project

### Requirements
+ Docker

If you don't have it, [get it](https://docs.docker.com/install/linux/docker-ce/ubuntu/)

### Instructions
Simply run:
```
./test.sh
```
