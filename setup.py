from setuptools import setup, find_packages
from codecs import open
from os import path

__author__ = 'Alisa Pavel'
__license__ = "GLP-3.0"
__email__ = ""


def get_requirements(remove_links=False):
    """
    lists the requirements to install.
    """

    try:
        with open('requirements.txt') as f:
            requirements = f.read().splitlines()
    except Exception as ex:
        with open('DecoraterBotUtils.egg-info\requires.txt') as f:
            requirements = f.read().splitlines()
    if remove_links:
        for requirement in requirements:
            # git repository url.
            if requirement.startswith("git+"):
                requirements.remove(requirement)
            # subversion repository url.
            if requirement.startswith("svn+"):
                requirements.remove(requirement)
            # mercurial repository url.
            if requirement.startswith("hg+"):
                requirements.remove(requirement)
    return requirements


here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(name='volta',
      version='0.1.1',
      license='',
      description='adVanced mOLecular neTwork Analysis',
      url='https://github.com/fhaive/VOLTA',
      author='Alisa Pavel',
      author_email='',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          # Pick your license as you wish (should match "license" above)
          'License :: OSI Approved :: BSD License',

          "Operating System :: POSIX :: Other",
          "Operating System :: Linux",

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          'Programming Language :: Python',
          'Programming Language :: Python :: 3'
      ],
      keywords='network-analysis, community-detection, complex-networks, network-clustering',
      install_requires=get_requirements(),
      long_description=long_description,
      long_description_content_type='text/markdown',
      extras_require={
       
      },
      dependency_links=[],
      packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test"]),
      )


