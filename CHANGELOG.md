# Changelog

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
