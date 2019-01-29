import filecmp
import glob
import os
import shutil
import tempfile
import unittest
import zipfile

from . import bundler

class TestBundler(unittest.TestCase):
    FIXTURES_DIR = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'fixtures'
    )
    TEST_MODEL_DIR = os.path.join(FIXTURES_DIR, 'test-model')
    TEST_TFLITE_FILE = os.path.join(FIXTURES_DIR, 'test.tflite')
    TEST_TFBUNDLE = os.path.join(FIXTURES_DIR, 'test.tfbundle')

    def setUp(self):
        self.output_directories = []

    def tearDown(self):
        for output_directory in self.output_directories:
            shutil.rmtree(output_directory)

    def create_temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.output_directories.append(temp_dir)
        return temp_dir

    def test_tflite_build_from_saved_model(self):
        outdir = self.create_temp_dir()
        tflite_file = os.path.join(outdir, 'model.tflite')
        bundler.tflite_build_from_saved_model(self.TEST_MODEL_DIR, tflite_file)
        self.assertTrue(filecmp.cmp(tflite_file, self.TEST_TFLITE_FILE))

    def test_tfbundle_build(self):
        outdir = self.create_temp_dir()
        outfile = os.path.join(outdir, 'test.tfbundle.zip')
        tfbundle_name = 'actual.tfbundle'
        bundler.tfbundle_build(
            os.path.join(self.TEST_TFBUNDLE, 'model.tflite'),
            os.path.join(self.TEST_TFBUNDLE, 'model.json'),
            os.path.join(self.TEST_TFBUNDLE, 'assets'),
            tfbundle_name,
            outfile
        )

        extraction_dir = self.create_temp_dir()
        with zipfile.ZipFile(outfile, 'r') as tfbundle_zip:
            tfbundle_zip.extractall(path=extraction_dir)

        extracted_paths_glob = os.path.join(extraction_dir, tfbundle_name, '**/*')
        extracted_paths = glob.glob(extracted_paths_glob, recursive=True)
        self.assertEqual(len(extracted_paths), 4)

        expected_files = {'model.tflite', 'model.json', 'assets', 'assets/labels.txt'}
        expected_paths = {
            os.path.join(extraction_dir, tfbundle_name, expected_file)
            for expected_file in expected_files
        }
        self.assertSetEqual(set(extracted_paths), expected_paths)
