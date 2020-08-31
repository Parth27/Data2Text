import setuptools

setuptools.setup(
    name="DataToText",
    version="0.0.1",
    author="Parth Diwanji",
    author_email="diwanji.parth@gmail.com",
    description="Deep Learning model for automated generation of text from structured data",
    url="https://github.com/Parth27/Data2Text",
    packages=setuptools.find_packages(),
    license='Apache License 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['pandas','numpy>=1.6.1','tensorflow-gpu==1.15','nltk','rouge','spacy','scipy>=0.9','scikit-learn'],
)