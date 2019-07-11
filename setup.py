#!/usr/bin/env python
from setuptools import setup,find_packages
import sys
from setuptools.command.test import test as TestCommand


class PyTest(TestCommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to pytest")]

    def initialize_options(self):
        TestCommand.initialize_options(self)
        self.pytest_args = ""

    def run_tests(self):
        # import here, cause outside the eggs aren't loaded
        import shlex
        import pytest
        self.pytest_args += " --cov=em_stitch --cov-report html "\
                            "--junitxml=test-reports/test.xml " \
                            "--ignore=src"

        errno = pytest.main(shlex.split(self.pytest_args))
        sys.exit(errno)

def opencv_mess():
    ver = None
    try:
        import cv2
        r = cv2.xfeatures2d
        ver = cv2.__version__
    except ImportError:
        # no opencv is installed, we don't need contrib
        required = 'opencv-python<=3.4.5'
    except AttributeError:
        # someone has opencv installed (but not contrib)
        # let's require some version
        required = 'opencv-python<=3.4.5'
    if ver:
        # opencv-contrib-python is installed
        # let's require some version
        required = 'opencv-contrib-python<3.4.3.0'
    return required


with open('test_requirements.txt', 'r') as f:
    test_required = f.read().splitlines()

with open('requirements.txt', 'r') as f:
    required = f.read().splitlines()
    required.append(opencv_mess())

setup(name='em_stitch',
      use_scm_version=True,
      description='a python package for stitching EM images',
      author_email='danielk@alleninstitute.org',
      url='https://github.com/AllenInstitute/em_stitch',
      packages=find_packages(),
      setup_requires=['setuptools_scm'],
      install_requires=required,
      tests_require=test_required,
      cmdclass={'test': PyTest})

