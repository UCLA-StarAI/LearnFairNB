from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import os

ext_modules = [
    Extension(
    name="pattern_finder.pfwrapper",
    sources=["pattern_finder/pf_wrapper.pyx",\
             "pattern_finder/pf_src/pattern_finder.cpp",\
             "pattern_finder/pf_src/search.cpp"],
    language="c++",
    extra_compile_args = "-ansi -fpermissive -Wall -O3 -ggdb -fPIC --std=c++11".split(),
    )
]

setup(
    name = 'pattern_finder',
    version = "1.0",
    packages = ['pattern_finder'],
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
)
