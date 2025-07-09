from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="proyecto_ia_rag_rl",
    version="0.2.0",
    description="Sistema de recomendación con RAG y RLHF para productos de bebés",
    author="Tu Nombre",
    author_email="tu@email.com",
    packages=find_packages(include=['src', 'demo']),
    package_dir={
        '': '.'
    },
    install_requires=requirements,
    python_requires='>=3.8',
    include_package_data=True,
    package_data={
        '': ['data/*', 'data/raw/*', 'data/processed/*', 'data/chroma_indexes/*'],
    },
    entry_points={
        'console_scripts': [
            'proyecto-ia=main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)