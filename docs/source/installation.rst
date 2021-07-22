Installation
============


Package dependencies
--------------------

Mandatory dependencies:

- tensorflow
- numpy
- scipy
- pandas
- h5py
- tqdm

Optional dependencies:

- refl1d:	only if fast generation of reflectivity curves is necessary (e.g. for training)
- matplotlib:	only if you want to see plots
- fabio:	only for parsing of ``.fio`` files



Installation via *pip*
----------------------

The package is mainly tested for Python 3.6, but there is a high probability that it might also work on
later version. It can simply be installed from the well-known PyPI repository via the *pip* tool *via*:

   ::    

    python3.6 -m pip install mlreflect

This will by default install the latest version of *mlreflect*, which is usually preferable due to
continued improvements, new features and bug fixes. However, **please observe whether the the package version
this document refers to is the same as the installed version** since between versions, changes to the API
might occur. For very recent versions there might not yet be a up-to-date documentation.


