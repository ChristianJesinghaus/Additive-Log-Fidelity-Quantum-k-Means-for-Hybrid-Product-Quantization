# Code Provenance and Licensing Notes

This document records the provenance of selected source-code portions and
clarifies the licensing of the current software distribution.

## 1. Current software license

The current source code, scripts, configuration files, and software
documentation in this repository are made available under the MIT License,
except where a file expressly states otherwise.

For project-specific material, this grant applies to the extent that Christian
Jesinghaus holds the applicable copyright or other rights.

The complete license text is available in:

- `LICENSE`
- `LICENSES/MIT.txt`

A research citation is appreciated but is not a condition of the MIT License.

## 2. Upstream classical PQKNN project

Parts of this repository are derived from the Product Quantization
k-Nearest Neighbors implementation by Jeroen Van Der Donckt:

- Author: Jeroen Van Der Donckt
- Upstream repository: https://github.com/jvdd/pqknn
- Reference revision:
  https://github.com/jvdd/pqknn/tree/4722d9be07bb12a9ba32e2fc9205dc54b8daf219
- Upstream MIT license:
  https://github.com/jvdd/pqknn/blob/4722d9be07bb12a9ba32e2fc9205dc54b8daf219/LICENSE

The upstream-derived material includes, among other portions:

- `product_quantization/PQKNN.py`
- the original basis of `product_quantization/__init__.py`
- `log_nb_clusters_to_np_int_type` in `product_quantization/util.py`
- `squared_euclidean_dist` in `product_quantization/util.py`

The current versions of those files and functions, including later
modifications made by Christian Jesinghaus, are distributed under the MIT
License.

The original upstream copyright and MIT notice are reproduced in:

`THIRD_PARTY_LICENSE/classical_PQKNN_MIT.txt`

## 3. Quantum-related and project-specific additions

The additional quantum-related helper implementations in
`product_quantization/util.py` include, among others:

- `amplitude_encoding`
- `quantum_minimum_finding_oracle`
- `quantum_grover_minimum_search`
- `quantum_parallel_cluster_assignment`
- `quantum_distance_calculation_parallel`
- `quantum_swap_test_distance`
- `quantum_superposition_centroid_update`
- `quantum_superposition_mean`
- `quantum_amplitude_estimation_distance`

These implementations were not intentionally adapted from a specific
third-party software implementation.

They were initially generated with the assistance of ChatGPT and were
subsequently reviewed by Christian Jesinghaus, tested in the context of the
project, integrated into the software, and in places manually modified.

The use of generative AI during development was also disclosed in the
associated Bachelor's thesis.

No separate third-party implementation or source-code version is identified
for these quantum helper functions because no specific external
implementation was intentionally used as their source.

To the extent that Christian Jesinghaus holds applicable rights in these
project-specific contributions, they are released under the MIT License.

## 4. Historical licensing note

Earlier published versions of `product_quantization/util.py` contained a
file-wide declaration:

`SPDX-License-Identifier: MIT`

This includes the following Git file-content versions discussed in the PQID
licensing enquiry:

- `30d2f918bdbb12b0ba59fbc099288dd38460387b`
- `d02646d3b93d95958dbd5aaa9ee13c30b4c0c8f4`

Those historical versions may be used under the MIT declaration that
accompanied them. The current MIT licensing of the software is consistent
with that historical declaration.

## 5. Research citation

If this software materially contributes to academic, scientific, or technical
work, users are requested to cite the Bachelor's thesis identified in
`CITATION.cff`.

This request supports scholarly attribution but is not an additional
condition of the MIT License.

## 6. Non-software material

The MIT software license does not automatically apply to PDFs, publication
manuscripts, theses, datasets, generated results, figures, reports, or model
artefacts. See `NOTICE` and any file-specific notices.
