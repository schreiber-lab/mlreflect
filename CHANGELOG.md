# Changelog

## [0.20.0] 2021-11-10

### Added

- Added more docstrings
- Added `simulate_reflectivity` flag for the `CurveFitter.fit_curve()` method that allows the skipping of 
  reflectivity simulation if prediction speed is of importance.
- Added `**kwargs` as an argument for the `TrainedModel` class that is passed directly to the method which loads the 
  Keras method to allow the loading of custom objects.

### Changed

- Replaced `scipy.optimize.minimize` with `scipy.optimize.curve_fit` because it's faster for least mean squares 
  minimization
- Changed docs to be compatible with Read the Docs. Now the docs can be found online (https://mlreflect.readthedocs.
  io/en/latest/).
- Updated tests.
- Refactored reflectivity interpolation to be all in one place
- Changed license from GPL-3 to MIT

### Fixed

- Fixed bug in minimizer interpolation

## [0.19.0] 2021-07-22

### Added

- Added new optimization method where the _q_ interpolation is sampled with small shifts in _q_ (around +-0.003 1/A).
  This option is turned on with the keyword `optimize_q=True` and generally improves the prediction.
- The relative fitting bounds for the LMS fit (when `polish=True`) can now be chosen for each of the three 
  prediction parameters (thickness, roughness, SLD) via the keyword `fraction_bounds`.
- Added log mean squared error to the `FitResult` object and the plot of the prediction.
- Added a lot more docstrings

### Changed

- The output of the `CurveFitter.fit_curve()` method has been changed from a tuple (containing reflectivity and 
  parameters) to a dictionary (now also including the best q shift if `optimize_q=True`).
- The scan number index of the `SpecParser.scan_info` dictionary was changed from `str` to `int` for consistency.
- Updated example notebooks
- Changed example SPEC file
- Updated documentation

## [0.18.0] 2021-07-14

### Added

- Added support for *.fio* files in the form of a `FioParser`, `FioLoader` and `FioFitter`. The code has been 
  refactored so that some functionality that is shared with the respective SPEC equivalents is inherited from a base
  class.
- Added a new subpackage `xrrloader.p08tools` that contains tools developed by Florian Bertram from DESY, Germany. 
  These tools help in reading the '.fio' files and reprocessing the corresponding 2D detector images.
- Added basic Sphinx docs to the project.

### Changed

- Updated some affected tests.
- Updated several docstrings to use syntax markdown for docs.
- Unified the version number from `setup.py` and `mlreflect.__init__` to `mlreflect.version`.

## [0.17.2] 2021-06-21

### Added

- Added SLD profile plotting to `FitResults` class.

### Fixed

- Fixed error in the `FitResult` class where the original curve was not plotted in the same range as the predicted 
  curve. 

## [0.17.1] 2021-06-10

### Added

- The `xrrloader` package is now included in the package code to avoid dependency problems.
- Added optional LMS fit refinement of the neural network prediction based on the log mean squared error of the 
  measured curve and predicted curve. To use this, the keyword argument `polish=True` has to be passed to the fit 
  methods of the `CurveFitter` and `SpecFitter` class.
- The `SpecFitter` class now provides its output as a `FitResult` object, which can be used to plot the prediction 
  or access different parts of the result more easily. The results plot now also shows the SLD profile of the 
  prediction.
- Added two example files as package resources, which are now detailed in `MANIFEST.in`:
  * `resources/examples/example.spec` is an example file that can be used to test `SpecFitter`.
  * `resources/examples/example.dat` is an ascii file that contains reflectivity data that can be used to test 
    `CurveFitter`.
- Complex layer parameters are now fully supported (in particular for complex SLDs).
- Added more tests.

### Changed

- The DefaultTrainedModel now has only 3 prediction parameters `Film_thickness`, `Film_roughness` and `Film_sld`,
  i.e. the thickness of the oxide layer is no longer predicted.
- The default model is now trained with absorption of the Si/SiOx substrate included.
- The number of weights of the DefaultTrainedModel has been reduced to save memory and prediction time.
- `CurveFitter.fit_curve()` can now also fit 2D arrays of data (instead of only single curves)
- Removed warning that was printed via print if the `refl1d` package is not available. Now it is `warnings.warn()` 
  and shows up only when the `ReflectivityGenerator` object is initialized.

### Fixed

- Fixed order of arguments of `SpecFitter.show_scans()`.
- Fixed wrong in-place behavior of `utils.label_helpers.convert_to_dataframe()`.

## [0.16.0] 2021-05-18

### Added

- Added an exception to the `ReflectivityGenerator.simulate_reflectivity()` method that allows a fall-back to the 
  built-in reflectivity simulation if for some reason the `refl1d` package cannot be loaded (for example if the C 
  code compilation failed).
- Added various docstrings for the `curve_fitter` module.
- Added `fit_range()` method to the `SpecFitter` class.
- Added more requirements to `requirements.txt` and added specific version numbers.

### Changed

- Made keyword arguments for `SpecFitter` class parameter methods explicit.
- Shift predicted `predicted_q_values` output from `SpecFitter` class by `dq` argument.

### Fixed

- Fixed test imports

## [0.15.2] 2021-04-30

### Added

- Added more docstrings for new classes.

### Changed

- Moved all high-level imports from the `mlreflect` package to its respective sub-packages for a better structure.

### Fixed

- Fixed dependency issues during where requirements are already called before they are installed
- Added `packaging` module to the requirements

## [0.15.1] 2021-04-30

### Added

- Added `UniformNoiseGenerator` class
- Added `Trainer` and `TrainedModel` classes to simplify on-the-fly training
- Added a `default_trained_model.h5` package resource that contains a trained keras model to fit XRR of a single 
  layer on a Si/SiOx substrate in air/vacuum.
- Added `DefaultTrainedModel` subclass that automatically loads the default model.
- Added `CurveFitter` and `SpecFitter` classes for easier on-the-fly fitting.

### Changed

- Update example notebook

### Fixed

- Fixed small bug in `mlreflect.noise.apply_scaling_factor()`. Now it returns a copy of the data.

## [0.15.0] 2021-04-27

### Added

- Added the `Parameter` class and its subclasses to make the definition of `Layer` objects easier.
- `Parameter` objects can also be sampled to create random labels for training data generation.
- Added a `Layer` subclass `ConstantLayer` which can only contain fixed parameters (no ranges)
- Added two `ConstantLayer` subclasses `Substrate` and `AmbientLayer` which are now used to make `MultilayerStructure`
  objects.
- `Substrate` objects don't have a thickness.
- `AmbientLayer` objects only have an SLD.
- A `MultilayerStructure` object can now be exported to and read from a dictionary with the `to_dict()` and
  `from_dict()` methods.
- Added a `copy()` method to all `Layer` and `MultilayerStructure` classes.

### Changed

- `Layer` objects now initialize with `Parameter` objects instead of range tuples for each thin film parameter.
- `MultilayerStructure` objects are no longer initialized with the ambient layer SLD. Instead, the ambient layer and 
  substrate are defined with separate methods.
- The substrate and ambient layer are by default constant layers
- Moved the different sampling distribution for training label generation to a new module
  `data_generation.distributions`
- The optional parameter `q_noise_spread` of the `ReflectivityGenerator` class to simulate noisy _q_ values was 
  moved from the initializer to the `simulate_reflectivity()` method.
- Removed the option to remove non-constant labels from the training labels when using the `OutputPreprocessor` class, 
  because it was somewhat confusing and found little use.

## [0.14.1] 2020-08-4

### Changed

- Updated all docstrings to that they can be read by the documentation generator Sphinx (using the Napoleon extension
 for Google-style docstrings).

## [0.14.0] 2020-07-31

### Added

- Added unit tests files `test_layers.py`, `test_data_generator.py`, `test_noise_generator.py` and
 `test_preprocessing.py` as well as the test runner `runner.py`.
- Added functions to h5_tools module that can be used to save the noise and background generated with the `noise`
 module to the save h5 file
- Added `NoiseGenerator` class to `training.noise_generator` which allows dynamic noise and background generation during
 training.
- Added `InputPreprocessor` properties `has_saved_standardization`, `standard_mean`, `standard_std`.
- Added `naming` module with `make_timestamp()` function to create identifiers for training output.
- Added `check_gpu.py` script to be able to quickly check if tensorflow can find the GPU.

### Changed

- Slightly changed the API of `OutputPreprocessor.restore_labels()`. It now only takes a single argument (the
 normalized labels). 
- Refactor internal package structure of the package source.
- Cleaned up a lot of attributes and methods and turned them into properties.
- Improved `has_saved_standardization` property of the `InputPreprocessor` class (return value depends now on whether
 or not `standard_mean` and `standard_std` are `True` or `False`.)
- Improved `__repr__` of `Layer` and `MultilayerStructure` classes.
- Updated `usage_example.ipynb` to work with the new API.
 
### Fixed

- Fixed several inconsistency bugs of the `label_removal_list` feature of the `OutputPreprocessor` class (with the
 help of unit tests).
 - Background and noise levels can now also be of `int` type (previously only `float`).

## [0.13.1] - 2020-06-03

### Added

- Added option for the `noise` module to generate shot noise and backgrounds with random levels within a given range.

### Changed

- The thickness of the bottom most layer (substrate) is no longer a label, because it has no influence on the data
 generation process and its presence was confusing.

### Fixed

- Fixed bug where the roughness of the bottom most layer was dependent on its thickness (which in turn was always set
 to 1). This led to a substrate roughness that was confined between 0 and 0.5 Å.

## [0.13.0] - 2020-04-22

### Added

- Added new C++-based reflectivity simulation engine from the refl1d package. This should be ~20 times faster than the
built-in code. The simulation engine can be chosen by using via the `engine` keyword of the ReflectivityGenerator class.
- Output normalization can now be changed from a [0, 1] range to a [-1, 1] range by choosing the approriate value for
the `normalization` keyword of the `OutputPreprocessor` class.
- Added method `InputPreprocessor.revert_standardization()`.

### Changed

- The ambient SLD is now also given as a range (instead of a single value) and can be used as a non-constant label for
training.
- `OutputPreprocessor.apply_preprocessing()` now returns a tuple containing preprocessed labels in addition to the
removed labels both as pandas DataFrames.
- The `removed_labels` DataFrame is now used to by the `restore_labels` method instead of the previous `training_labels`
DataFrame (which caused some confusion).
- Removed methods and properties too reduce overhead and limit inconsistencies.
- Changed the .h5 file format that is with the `h5_tools` module.
    - It is now no longer designed to save training, validation and testing data in separate groups to give the user
    more flexibility. As a result, the group hierarchy was reduced by one level ("data" group was removed).
    - All non-data information (units, min/max label values, etc.) have now been moved to the "info" group.
- Removed job list functionality from the `InputPreprocessor` class because it was unintuitive to use. Now the class is
only used for input standardization.
- Moved methods `apply_shot_noise()`, `generate_background()`, `apply_gaussian_convolution()` of the
`ReflectivityGenerator` class to a new `noise` module as stand-alone functions  that can be applied to any previously
generated reflectivity curves.
- `apply_gaussian_convolution()` now uses the gaussian convolution from the refl1d package.
- Updated example notebook `data/notebooks/usage_example.ipynb` to match API of version 0.13.0

### Fixed

- Non-constant labels that are removed via the `OutputPreprocessor` class are now not incorrectly added to the restored
labels anymore.
- Fixed that the wrong number of thin film layers was saved to the .h5 file when using the `h5_tools` module.
- The built-in reflectivity engine can now no longer generate intensities that are higher than 1.

## [0.12.2] - 2020-04-08

### Changed

- Changed all instances of keras to tensorflow.keras

## [0.12.1] - 2020-01-22

### Added

- Added CHANGELOG.md file to the project to track changes between releases
- Added method MultilayerStructure.rename_layer()
- Added docstrings to MultilayerStructure class

### Changed
- SimpleModel now returns tuple (hist, timestamp) instead of only hist

## [0.12.0] - 2020-01-17

### Added

- Samples are now defined via Layer and MultilayerStructure objects
- This allows for more intuitive building of defined sample structures

### Changed

- Unified the layer order that different parts of the API expect (it's now always from bottom to top)

## [0.11.1] - 2020-01-17

### Added

- mlreflect can now be installed directly from PyPI via pip (refer to README file)
- Added usage example section to README.md

## [0.11.0] - 2020-01-16

### Added

- SLD profile generation
- Added option to apply Gaussian blur to reflectivity curves
- Added version number to package
- Added ability to choose 'bolstered' or 'uniform' distribution when generating labels
- Added install instructions for pip to the README file

### Changed

- Improved Jupyter notebook tutorial
- Improve the way models are saved when using model_helpers
- Allow noise options to be passed when initializing ReflectivityGenerator

### Fixed

- Fixed small scaling error in shot noise scaling

## [0.10.0] - 2019-11-07

* First public release of the mlreflect update
