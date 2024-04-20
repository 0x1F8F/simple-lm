from setuptools import setup, find_packages

# setuptools.config.


setup(
    name="simple-bigram",
    version="0.1.0",
    description="simple transformer model",
    long_description="simple transformer model",
    author="",
    author_email="",
    maintainer="",
    maintainer_email="",
    url="",
    download_url="",
    requires=[
        "torch",
    ],

    packages=find_packages(
        where=".", 
        exclude=["data", "tmp", "simple_bigram.test"],
        include= [ "simple_bigram"]
    ),
    
    entry_points = {
        'console_scripts' : [
            'bigram = simple_bigram:main'
        ]
    }
)
