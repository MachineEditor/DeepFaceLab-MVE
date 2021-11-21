# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.8.0] - 2021-06-20
### Added
- Morph factor option
- Migrated options from SAEHD to AMP:
  - Loss function
  - Random downsample
  - Random noise
  - Random blur
  - Random jpeg
  - Background Power
  - CT mode: fs-aug
  - Random color

## [1.7.3] - 2021-06-16
### Fixed
- AMP mask type

## [1.7.2] - 2021-06-15
### Added
- New sample degradation options (only affects input, similar to random warp): 
  - Random noise (gaussian/laplace/poisson)
  - Random blur (gaussian/motion)
  - Random jpeg compression
  - Random downsampling
- New "warped" preview(s): Shows the input samples with any/all distortions.     

## [1.7.1] - 2021-06-15
### Added
- New autobackup options:
  - Session name
  - ISO Timestamps (instead of numbered)
  - Max number of backups to keep (use "0" for unlimited)

## [1.7.0] - 2021-06-15
### Updated
- Merged in latest changes from upstream, including new AMP model

## [1.6.2] - 2021-05-08
### Fixed
- Fixed bug with GAN smoothing/noisy labels with certain versions of Tensorflow

## [1.6.1] - 2021-05-04
### Fixed
- Fixed bug when `fs-aug` used on model with same resolution as dataset

## [1.6.0] - 2021-05-04
### Added
- New loss function "MS-SSIM+L1", based on ["Loss Functions for Image Restoration with Neural Networks"](https://research.nvidia.com/publication/loss-functions-image-restoration-neural-networks)

## [1.5.1] - 2021-04-23
### Fixed
- Fixes bug with MS-SSIM when using a version of tensorflow < 1.14

## [1.5.0] - 2021-03-29
### Changed
- Web UI previews now show preview pane as PNG (loss-less), instead of JPG (lossy), so we can see the same output 
  as on desktop, without any changes from JPG compression. This has the side-effect of the preview images loading slower
  over web, as they are now larger, a future update may be considered which would give the option to view as JPG 
  instead.

## [1.4.2] - 2021-03-26
### Fixed 
- Fixes bug in background power with MS-SSIM, that misattributed loss from dst to src

## [1.4.1] - 2021-03-25
### Fixed
- When both Background Power and MS-SSIM were enabled, the src and dst losses were being overwritten with the 
  "background power" losses. Fixed so "background power" losses are properly added with the total losses.
    - *Note: since all the other losses were being skipped when ms-ssim and background loss were being enabled, this had 
      the side-effect of lowering the memory requirements (and raising the max batch size). With this fix, you may 
      experience an OOM error on models ran with both these features enabled. I may revisit this in another feature, 
      allowing you to manually disable certain loss calculations, for similar performance benefits.*

## [1.4.0] - 2021-03-24
### Added
- [MS-SSIM loss training option](doc/features/ms-ssim)
- GAN version option (v2 - late 2020 or v3 - current GAN)
- [GAN label smoothing and label noise options](doc/features/gan-options)
### Fixed
- Background Power now uses the entire image, not just the area outside of the mask for comparison.
This should help with rough areas directly next to the mask

## [1.3.0] - 2021-03-20
### Added
- [Background Power training option](doc/features/background-power/README.md)

## [1.2.1] - 2021-03-20
### Fixed
- Fixes bug with `fs-aug` color mode.

## [1.2.0] - 2021-03-17
### Added
- [Random color training option](doc/features/random-color/README.md)

## [1.1.5] - 2021-03-16
### Fixed
- Fixed unclosed websocket in Web UI client when exiting

## [1.1.4] - 2021-03-16
### Fixed
- Fixed bug when exiting from Web UI

## [1.1.3] - 2021-03-16
### Changed
- Updated changelog with unreleased features, links to working branches

## [1.1.2] - 2021-03-12
### Fixed
- [Fixed missing predicted src mask in 'SAEHD masked' preview](doc/fixes/predicted_src_mask/README.md)

## [1.1.1] - 2021-03-12
### Added
- CHANGELOG file for tracking updates, new features, and bug fixes
- Documentation for Web UI
- Link to CHANGELOG at top of README

## [1.1.0] - 2021-03-11
### Added
- [Web UI for training preview](doc/features/webui/README.md)

## [1.0.0] - 2021-03-09
### Initialized
- Reset stale master branch to [seranus/DeepFaceLab](https://github.com/seranus/DeepFaceLab), 
  21 commits ahead of [iperov/DeepFaceLab](https://github.com/iperov/DeepFaceLab) ([compare](https://github.com/iperov/DeepFaceLab/compare/4818183...seranus:3f5ae05))

[1.8.0]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.7.3...v1.8.0
[1.7.3]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.7.2...v1.7.3
[1.7.2]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.7.1...v1.7.2
[1.7.1]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.6.2...v1.7.0
[1.6.2]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.6.1...v1.6.2
[1.6.1]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.5.1...v1.6.0
[1.5.1]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.4.2...v1.5.0
[1.4.2]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.4.1...v1.4.2
[1.4.1]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.4.0...v1.4.1
[1.4.0]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.2.0...v1.2.1
[1.2.0]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.5...v1.2.0
[1.1.5]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.4...v1.1.5
[1.1.4]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.3...v1.1.4
[1.1.3]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.2...v1.1.3
[1.1.2]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.1...v1.1.2
[1.1.1]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.1.0...v1.1.1
[1.1.0]: https://github.com/faceshiftlabs/DeepFaceLab/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/faceshiftlabs/DeepFaceLab/releases/tag/v1.0.0
