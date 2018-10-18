# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='emotrix',
    version='0.0.2',
    description='BCI api for handling human emotions',
    long_description=readme,
    author='Henzer G., César L., Freddy M., Matías V., Pablo S., Jackeline G., Angel M., Mario B., Kevin G., Sergio G., Diego J., Elisa P., Nancy M. and Maryalis.',
    author_email='',
    url='https://github.com/emotrix/EMOTRIX',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
