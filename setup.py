from distutils.core import setup
import Cython
from Cython.Build import cythonize
from distutils.extension import Extension
Cython.Compiler.Options.annotate = True
setup(
    name='gym_watten',
    ext_modules = cythonize([Extension("gym_watten.envs.watten_env", ["gym_watten/envs/watten_env.pyx"], language="c++")]),
    packages=['gym_watten']
)