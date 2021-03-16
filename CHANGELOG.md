# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]
### Added
- [Random color training option](https://github.com/faceshiftlabs/DeepFaceLab/tree/feature/random-color)
- [MS-SSIM loss training option](https://github.com/faceshiftlabs/DeepFaceLab/tree/feature/ms-ssim-loss-2)
### In Progress
- [Freezeable layers (encoder/decoder/etc.)](https://github.com/faceshiftlabs/DeepFaceLab/tree/feature/freezable-weights)

## [1.1.5] - 2020-03-16
### Fixed
- Fixed unclosed websocket in Web UI client when exiting

## [1.1.4] - 2020-03-16
### Fixed
- Fixed bug when exiting from Web UI

## [1.1.3] - 2020-03-16
### Changed
- Updated changelog with unreleased features, links to working branches

## [1.1.2] - 2020-03-12
### Fixed
- [Fixed missing predicted src mask in 'SAEHD masked' preview](doc/fixes/predicted_src_mask/README.md)

## [1.1.1] - 2020-03-12
### Added
- CHANGELOG file for tracking updates, new features, and bug fixes
- Documentation for Web UI
- Link to CHANGELOG at top of README

## [1.1.0] - 2020-03-11
### Added
- [Web UI for training preview](doc/features/webui/README.md)

## [1.0.0] - 2021-03-09
### Initialized
- Reset stale master branch to [seranus/DeepFaceLab](https://github.com/seranus/DeepFaceLab), 
  21 commits ahead of [iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab) ([compare](https://github.com/iperov/DeepFaceLab/compare/4818183...seranus:3f5ae05))

[Unreleased]: https://github.com/olivierlacan/keep-a-changelog/compare/v1.1.5...HEAD
[1.1.5]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.4...v1.1.5
[1.1.4]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/faceshiftlabs/DeepFaceLab/releases/tag/v1.0.0
