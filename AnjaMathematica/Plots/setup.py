from setuptools import setup, Extension
from Cython.Build import cythonize

# Definiere die Erweiterung
ext_modules = [
    Extension(
        "function",
        sources=["function.pyx"],  
        extra_compile_args=["-std=c++11", "-stdlib=libc++"],  # Hinzufügen von libc++
        language="c++"
    )
]

# Führe das Setup-Skript aus
setup(
    ext_modules=cythonize(ext_modules)
)
