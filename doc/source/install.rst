.. Testing cross-references:

.. _installation:

Installing the cgptoolbox
=========================

.. highlight:: bash

.. This is a comment. In fact, any directive that ReST cannot make sense of
   is considered a comment. Furthermore, ReST is picky about indentation. 
   In the enumerated lists below, the indentation of continuation lines must 
   exactly match the beginning of the line, not counting the number and dot. 
   There must be a blank line before further paragraphs in an item, and before 
   nested lists.

.. todo::

   Describe non-python dependencies (R, hdf5, zeromq, lxml).
   Register package at PyPI:
   http://guide.python-distribute.org/contributing.html#registering-projects

Installing on `Ubuntu <http://www.ubuntu.com/download/ubuntu/download>`_ 12.10 with root access
-----------------------------------------------------------------------------------------------

Installing cgptoolbox on the latest version of Ubuntu is relatively easy and 
Ubuntu also works well under `VirtualBox <http://www.virtualbox.org/>`_ on 
64-bit Windows 7 and other platforms. Most dependencies such as `hdf5 
<http://www.hdfgroup.org/HDF5/>`_, R_ and various 
`Python <http://python.org>`_ packages are in the Ubuntu repositories and can 
be installed with apt-get. The code line will install all dependencies.

.. code-block:: bash

   sudo apt-get install python-tables python-networkx python-pip python-rpy2 python-joblib cython

For solving differential equations the cgptoolbox utilizes the `SUNDIALS 
<http://www.llnl.gov/CASC/sundials>`_ library and the python bindings provided 
by the Python package `pysundials <http://pysundials.sourceforge.net>`_. 
Ubuntu 12.10 offer SUNDIALS v.2.5.0, but the stable release of `pysundials 
<http://pysundials.sourceforge.net>`_ is for v.2.3.0. The following lines of 
code adds the repository and signing key for Debian stable, installs libraries 
and header files for the 2.3.0 version of SUNDIALS and protect the packages 
from upgrading.

.. code-block:: bash

   sudo add-apt-repository 'deb http://ftp.de.debian.org/debian squeeze main'
   sudo apt-get install debian-archive-keyring					#not tested properly yet
   sudo apt-get install libsundials-serial=2.3.0-2 libsundials-serial-dev=2.3.0-2
   echo "libsundials-serial hold" | sudo dpkg --set-selections
   echo "libsundials-serial-dev hold" | sudo dpkg --set-selections

Once the right version of SUNDIALS is installed, the cgptoolbox can 
be installed with `pip <http://www.pip-installer.org/>`_ directly from the 
code repository.

.. code-block:: bash

   sudo pip install --process-dependency-links git+https://github.com/jonovik/cgptoolbox.git
   
Installing on Linux witout root access 
--------------------------------------

If you don't have root access, e.g. on a shared high-performance computing 
cluster, you must either get a system administrator to install dependencies, or 
install them under your home directory. For the latter, please see the
:ref:`environment-variables` section.

Dependencies (version numbers refer to versions that we have tested; other 
versions might also work):

* `Python <http://python.org>`_  versions 2.7.3, 2.7.2. Python must be 
  installed with header files. There are several convenient 
  `scientific Python distributions 
  <http://stackoverflow.com/questions/6719309/python-distributions-and-environments-for-scientific-computing>`_.
* `virtualenv <http://www.virtualenv.org>`_ version 1.8.2
* `R <http://www.r-project.org/>`_ , version 2.15.1. R must be built as a library (instructions :ref:`below <r-instructions>`).
* `hdf5 <http://www.hdfgroup.org/HDF5/>`_ , version 1.8.7, 1.8.9.
* `SUNDIALS <http://www.llnl.gov/CASC/sundials>`_ 2.3.0 built as shared library (instructions :ref:`below <sundials-instructions>`)
* `lxml <http://lxml.de>`_ and its dependencies ``libxml2`` and ``libxslt``.
  If these are not installed, you could try setting the environment variable 
  STATIC_DEPS=true and proceed as described in the 
  `lxml installation instructions <http://lxml.de/installation.html>`_, 
  e.g. ``STATIC_DEPS=true pip install lxml``. If this fails, install the 
  dependencies manually and make sure to adjust LD_LIBRARY_PATH as described 
  below.
* Note on `matplotlib <http://matplotlib.org>`_: For a headless installation
  (i.e. one without a graphical display), you may wish to set ``backend: Agg``
  in your `matplotlibrc 
  <http://matplotlib.org/users/customizing.html#the-matplotlibrc-file>`_ file.

.. environment-variables:

Environment variables
^^^^^^^^^^^^^^^^^^^^^

The examples below install libraries under ``$HOME/usr``, in which case you 
should put the following in your `bash startup file 
<http://www.gnu.org/software/bash/manual/html_node/Bash-Startup-Files.html>`_
(e.g. ``~/.bash_profile``)::

   export PATH=$HOME/usr/bin:$PATH
   export CPATH=$HOME/usr/include:$CPATH
   export LD_LIBRARY_PATH=$HOME/usr/lib:$LD_LIBRARY_PATH

You may need to add additional directories to PATH, CPATH and LD_LIBRARY_PATH, 
depending on how you have installed e.g. your Python distribution; see the 
respective documentation on how to put the Python executable on PATH 
and header files in CPATH and friends.

A useful option for the ``pip`` package installer for Python is to cache 
downloaded files::

   export PIP_DOWNLOAD_CACHE=$HOME/.pip-cache

Also (this is primarily useful if you use ``pip`` *without* ``virtualenv``), 
you can install packages to a non-root location by setting::

   export PIP_INSTALL_OPTION=--prefix=$HOME/usr

.. _sundials-instructions:

SUNDIALS 
^^^^^^^^

Download version 2.3.0 of the `SUNDIALS <http://www.llnl.gov/CASC/sundials>`_ 
library not the newest 2.5.0, since `pysundials 
<http://pysundials.sourceforge.net>`_ does not work for the latest version yet.

.. code-block:: bash

   tar -xzf sundials-2.3.0.tar.gz
   cd sundials-2.3.0
   ./configure --prefix=$HOME/usr --enable-shared
   make
   make install
   cd ..

rpy2
^^^^^^^^^^^^^^^^^^^^

See http://rpy.sourceforge.net/rpy2/doc-2.3/html/overview.html#installation.

In short, install R as a shared library (described in the next paragraph), 
then install ``rpy2`` e.g. using ``pip``. If installation fails, try a newer 
revision of ``rpy2``. On one cluster, 2.3.1 failed but revision
`6d055a3909e9 <https://bitbucket.org/lgautier/rpy2/commits/6d055a3909e9>`_
succeded.

.. _r-instructions:

R - build as library
""""""""""""""""""""
.. code-block:: bash

   #download tarball from mirror and extract
   wget http://cran.uib.no/src/base/R-2/R-2.15.2.tar.gz
   tar xzf R-2.15.2.tar.gz						

   #configure, compile and install
   cd R-2.15.2
   ./configure --prefix=$HOME/usr --enable-R-shlib
   make
   make install
   cd ..

Virtualenv with required python packages
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   #create and activate virtual Python environment
   virtualenv cgp			
   source cgp/bin/activate
   pip install --process-dependency-links git+https://github.com/jonovik/cgptoolbox.git

..  Unfinished draft:
    Example: Install with minimal use of root on Ubuntu
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    
    Tested on a fresh install of Ubuntu 12.04 LTS. This assumes that you have 
    somehow installed Subversion, Git, and R (the equivalent of Ubuntu packages 
    ``subversion git r-base-dev``).
    
    * Edit :ref:`environment-variables` in ``~/.bashrc``.
    * Install EPD, specify $HOME/usr as installation directory.
    * Install :ref:`sundials-instructions`.
    * Run the following commands in the terminal. The ``--system-site-packages`` 
      option makes the EPD modules (numpy, lxml et al.) available in the virtual 
      environment::
      
      easy_install virtualenv
      virtualenv --system-site-packages ~/venv/cgp
      source ~/venv/cgp/bin/activate
      pip install --process-dependency-links git+https://github.com/jonovik/cgptoolbox.git

Testing
-------

To test if lxml, pysundials, rpy2 and their respective dependencies are 
properly installed::

   python -c "from lxml import etree"
   python -c "from pysundials import cvode"
   python -c "from rpy2 import rinterface"

To run all unit tests for the cgptoolbox, checkout the source code, change to the cgptoolbox directory, and run::

   nosetests cgp

This will run ``nose`` with the options specified in :download:`setup.cfg <../../setup.cfg>`.

Building the documentation
--------------------------

The documentation is written using `Sphinx <http://sphinx.pocoo.org/latest/>`_. 
Once the cgptoolbox is available on 
`PyPI <http://pypi.python.org/pypi>`_, the following should work:

   pip install cgptoolbox[docs]

Unfortunately, URLs cannot be combined with the [] for specifying extras to 
install. As a workaround, check out the source, then install:

   git clone https://github.com/jonovik/cgptoolbox.git
   cd cgptoolbox
   pip install -e .[docs]

Then run the following from the ``cgptoolbox/doc`` directory::

   sphinx-apidoc -T -H cgptoolbox -o source ..
   make html

Documentation will end up in ``cgptoolbox/doc/build/html``.

Here's a one-liner to make both html and latex, ignoring any latex errors::

   time ((make clean; make html; make latex; cd build/latex; echo R | pdflatex cgptoolbox.tex; echo R | pdflatex cgptoolbox.tex; xdg-open cgptoolbox.pdf; xdg-open ../html/index.html) > all.txt 2>&1)

.. rubric:: If a plain ``make html`` fails:

For a pristine build, you can try wiping all doc files that are 
not under version control.

.. warning::
   
   You will probably want a dry run first to see 
   what will be removed. Run this from your ``cgptoolbox/doc`` directory::
   
      git clean -n -d -x
   
   Then, if you are certain that no important work will be lost::
   
      git clean -f -d -x
      sphinx-apidoc -T -H cgptoolbox -o source ..
      make html

Please report errors or omissions to jonovik@gmail.com.
