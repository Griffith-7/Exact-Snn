# Changelog

## 2.0.0 - 2026-09-03

- Added PyTorch-native exact-gradient TTFS, convolutional, multi-spike, and recurrent layers.
- Added optional event-driven, existence-gradient, normalization, reset-reference, initializer, loss, and spike-time utility modules.
- Added firing calibration for TTFS layers and networks.
- Added finite-difference, serialization, dtype, and companion-module tests.
- Added package metadata, release documentation, and CI configuration.

## Versioning policy

This project follows semantic versioning. Public imports, class names, function names,
and documented tensor shape contracts are treated as the public API.

- Patch releases fix bugs without changing public contracts.
- Minor releases add backward-compatible layers, losses, utilities, or options.
- Major releases may change public behavior or remove deprecated APIs.

The mathematical implementation may improve between releases, but changes to numerical
accuracy or firing behavior will be documented in the release notes.
