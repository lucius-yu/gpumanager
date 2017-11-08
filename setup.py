from setuptools import setup
import gpumanager

def read_readme():
    with open('README.md') as f:
        return f.read()

setup(
    name='gpumanager',
    version=gpumanager.__version__,
    license='MIT',
    description='An utility to choose NVIDIA GPU',
    long_description=read_readme(),
    url='https://github.com/eyulush/gpumanager',
    author='Lucius Yu',
    author_email='eyulush@gmail.com',
    keywords='nvidia-smi choose gpu',
    classifiers=[
        # https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Topic :: System :: Monitoring',
    ],
    #packages=['gpumanager'],
    py_modules=['gpumanager'],
    install_requires=[
    ],
)
