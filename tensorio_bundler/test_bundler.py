import filecmp
import os
import shutil
import tempfile
import unittest

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
        print(self.TEST_MODEL_DIR)
        self.output_directories = []

    def tearDown(self):
        for output_directory in self.output_directories:
            shutil.rmtree(output_directory)

    def create_temp_dir(self):
        temp_dir = tempfile.mkdtemp()
        self.output_directories.append(temp_dir)
        return temp_dir

    def test_toco_build_from_saved_model(self):
        outdir = self.create_temp_dir()
        tflite_file = os.path.join(outdir, 'model.tflite')
        bundler.tflite_build_from_saved_model(self.TEST_MODEL_DIR, tflite_file)
        self.assertTrue(filecmp.cmp(tflite_file, self.TEST_TFLITE_FILE, shallow=False))

    def test_tfbundle_build(self):
        outdir = self.create_temp_dir()
        outfile = os.path.join(outdir, 'test.tfbundle.zip')
        bundler.tfbundle_build(
            os.path.join(self.TEST_TFBUNDLE, 'model.tflite'),
            os.path.join(self.TEST_TFBUNDLE, 'model.json'),
            os.path.join(self.TEST_TFBUNDLE, 'assets'),
            'actual.tfbundle',
            outfile
        )
