from setuptools import setup, find_packages

setup(
    name='flash', 
    version='0.1.0',
    author='Althaf Muhammad',
    author_email='zoory9900@gmail.com',
    description='A package for data science and machine learning utilities.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/flash-lib/flash',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'scipy'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
    test_suite='tests',
    include_package_data=True,
    zip_safe=False,
)
