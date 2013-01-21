import multitask
from distutils.core import setup

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix

"""

setup(
    name='minitrace',
    description='A module for machine learning models with trace norm penalties',
    long_description=open('README.rst').read(),
    version=multitask.__version__,
    author='Fabian Pedregosa',
    author_email='fabian@fseoane.net',
    url='http://pypi.python.org/pypi/minitrace',
    py_modules=['multitask'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    license='Simplified BSD'

)