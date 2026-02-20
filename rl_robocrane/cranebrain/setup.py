# setup.py
from setuptools import setup, find_packages

setup(
    name='cranebrain',                  # Your package name
    version='0.1.0',                    # Version of your package
    description='MPC formulation for the robocrane system based on acados and much more!',
    author='Gerald Ebmer',
    author_email='gerald.ebmer@tuwien.ac.at',
    url='https://github.com/yourusername/my_package',  # Your package repository (optional)
    packages=find_packages(),            # Automatically finds your package modules
    install_requires=[],                 # List of dependencies (if any)
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',            # Python version requirement
)