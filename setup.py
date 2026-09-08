from setuptools import setup, find_packages

setup(
    name="signature_diffusion_osv",
    version="1.0.0",
    description="Neuromotor Kinematic and Latent Diffusion Framework for Offline Signature Verification",
    author="Anonymous Authors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "diffusers>=0.25.0",
        "transformers>=4.35.0",
        "timm>=0.9.12",
        "opencv-python>=4.8.0",
        "scikit-image>=0.20.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
