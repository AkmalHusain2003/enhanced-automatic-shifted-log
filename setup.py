#!/usr/bin/env python
"""
Setup script for Enhanced Automatic Shifted Log Transformer

This setup.py provides backward compatibility alongside pyproject.toml.
For modern installations, pyproject.toml is preferred.

Author: Muhammad Akmal Husain
Email: akmalhusain2003@gmail.com
"""

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

# Ensure minimum Python version
if sys.version_info < (3, 7):
    sys.exit("Python 3.7 or higher is required. You are using Python {}.{}".format(*sys.version_info[:2]))

# Get the long description from the README file
here = Path(__file__).parent.resolve()
long_description = (here / "README.md").read_text(encoding="utf-8")

# Read version from __init__.py to maintain single source of truth
def get_version():
    """Extract version from __init__.py"""
    version_file = here / "enhanced_aslt" / "__init__.py"
    if version_file.exists():
        with open(version_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.startswith('__version__'):
                    # Extract version string
                    version = line.split('=')[1].strip().strip('"').strip("'")
                    return version
    return "0.1.0"  # Fallback version

# Define requirements
install_requires = [
    "numpy>=1.19.0",
    "pandas>=1.3.0", 
    "scikit-learn>=1.0.0",
    "scipy>=1.7.0",
    "numba>=0.50.0"
]

# Development dependencies
dev_requires = [
    "pytest>=6.0",
    "pytest-cov>=2.10",
    "pytest-xdist>=2.2",
    "black>=21.0",
    "flake8>=3.9", 
    "isort>=5.9",
    "mypy>=0.910",
    "pre-commit>=2.15",
    "jupyter>=1.0.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0"
]

# Test dependencies
test_requires = [
    "pytest>=6.0",
    "pytest-cov>=2.10",
    "pytest-xdist>=2.2", 
    "hypothesis>=6.0"
]

# Documentation dependencies
docs_requires = [
    "sphinx>=4.0",
    "sphinx-rtd-theme>=1.0",
    "numpydoc>=1.1",
    "myst-parser>=0.15"
]

# Example dependencies
examples_requires = [
    "jupyter>=1.0.0",
    "matplotlib>=3.3.0",
    "seaborn>=0.11.0",
    "plotly>=5.0.0"
]

# Performance dependencies
performance_requires = [
    "numba>=0.56.0",
    "llvmlite>=0.39.0"
]

# All extra dependencies combined
all_requires = dev_requires + test_requires + docs_requires + examples_requires + performance_requires

# Remove duplicates while preserving order
all_requires = list(dict.fromkeys(all_requires))

# Package configuration
setup(
    # Basic package information
    name="enhanced-automatic-shifted-log",
    version=get_version(),
    description="Enhanced Automatic Shifted Log Transformer with Monte Carlo optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author and contact information
    author="Muhammad Akmal Husain",
    author_email="akmalhusain2003@gmail.com",
    maintainer="Muhammad Akmal Husain", 
    maintainer_email="akmalhusain2003@gmail.com",
    
    # URLs and links
    url="https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log",
    download_url="https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log/archive/v0.1.0.tar.gz",
    project_urls={
        "Homepage": "https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log",
        "Repository": "https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log.git",
        "Documentation": "https://enhanced-automatic-shifted-log.readthedocs.io",
        "Changelog": "https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log/blob/main/CHANGELOG.md",
        "Issues": "https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log/issues",
        "Bug Reports": "https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log/issues",
        "Source": "https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log",
    },
    
    # License
    license="MIT",
    
    # Package discovery and inclusion
    packages=find_packages(exclude=["tests", "tests.*", "docs", "docs.*", "examples"]),
    package_dir={"": "."},
    
    # Include additional files
    include_package_data=True,
    package_data={
        "enhanced_aslt": ["py.typed", "*.pyi"],
    },
    
    # Dependencies
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "dev": dev_requires,
        "test": test_requires, 
        "docs": docs_requires,
        "examples": examples_requires,
        "performance": performance_requires,
        "all": all_requires,
    },
    
    # Entry points for command line tools
    entry_points={
        "console_scripts": [
            "aslt-benchmark=enhanced_aslt.cli:benchmark_command",
            "aslt-validate=enhanced_aslt.cli:validate_command",
        ],
    },
    
    # Classification and metadata
    classifiers=[
        # Development status
        "Development Status :: 4 - Beta",
        
        # Intended audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        
        # License
        "License :: OSI Approved :: MIT License",
        
        # Operating systems
        "Operating System :: OS Independent",
        "Operating System :: POSIX",
        "Operating System :: Unix", 
        "Operating System :: MacOS",
        "Operating System :: Microsoft :: Windows",
        
        # Programming language
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: Implementation :: CPython",
        
        # Topics
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Mathematics", 
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        "Topic :: Utilities",
    ],
    
    # Keywords for package discovery
    keywords=[
        "data-transformation",
        "log-transformation",
        "normality", 
        "monte-carlo-optimization",
        "statistical-preprocessing",
        "data-science",
        "machine-learning",
        "numba-acceleration",
        "feng-transformation", 
        "adaptive-algorithms",
        "skewness-correction",
        "data-normalization",
        "outlier-handling",
        "robust-statistics"
    ],
    
    # Zip safety
    zip_safe=False,
    
    # Setup requires for build dependencies
    setup_requires=[
        "setuptools>=61.0",
        "wheel",
        "numpy>=1.19.0",  # Required for numba compilation
    ],
    
    # Tests configuration
    test_suite="tests",
    tests_require=test_requires,
    
    # Additional options
    options={
        "bdist_wheel": {
            "universal": False,  # Not universal because of numba
        },
        "build_ext": {
            "inplace": True,
        }
    },
    
    # Command classes for custom build behavior
    cmdclass={},
    
    # Platform specific configurations
    platforms=["any"],
)

# Post-installation message
def print_success_message():
    """Print success message after installation"""
    print("\n" + "="*80)
    print("Enhanced Automatic Shifted Log Transformer installed successfully!")
    print("="*80)
    print("Quick start:")
    print("  >>> from enhanced_aslt import AutomaticShiftedLogTransformer")
    print("  >>> transformer = AutomaticShiftedLogTransformer()")
    print("  >>> transformer.fit(your_data)")
    print("  >>> transformed_data = transformer.transform(your_data)")
    print("")
    print("Documentation: https://enhanced-automatic-shifted-log.readthedocs.io")
    print("Issues: https://github.com/AkmalHusain2003/enhanced-automatic-shifted-log/issues")
    print("="*80)

# Custom command to show post-install message
try:
    from setuptools.command.install import install
    from setuptools.command.develop import develop
    
    class PostInstallCommand(install):
        """Post-installation for installation mode."""
        def run(self):
            install.run(self)
            print_success_message()
    
    class PostDevelopCommand(develop):
        """Post-installation for development mode."""
        def run(self):
            develop.run(self)
            print_success_message()
    
    # Update cmdclass with custom commands
    setup.cmdclass.update({
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    })
    
except ImportError:
    # If setuptools doesn't support these commands, continue without them
    pass

if __name__ == "__main__":
    # Check if we're running in a supported environment
    if os.environ.get('NUMBA_DISABLE_JIT'):
        print("Warning: NUMBA_DISABLE_JIT is set. Performance may be reduced.")
    
    # Additional environment checks
    try:
        import numpy
        print(f"NumPy version: {numpy.__version__}")
    except ImportError:
        print("Warning: NumPy not found. It will be installed as a dependency.")
    
    try:
        import numba
        print(f"Numba version: {numba.__version__}")
    except ImportError:
        print("Numba will be installed as a dependency for performance acceleration.")
    
    # Run setup
    setup()