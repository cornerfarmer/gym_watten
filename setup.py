from distutils.core import setup
import Cython
from Cython.Build import cythonize
from distutils.extension import Extension
Cython.Compiler.Options.annotate = True
setup(
    ext_modules = cythonize([Extension("watten_env", ["envs/watten_env.pyx"], language="c++")])
)
