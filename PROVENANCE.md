# Code Provenance and Licensing Notes

This document records the provenance of selected source-code portions in this
repository and clarifies the licensing relationship between upstream-derived
code, later project-specific additions, and AI-assisted code generation.

## 1. Upstream classical PQKNN code

Parts of this repository are derived from the open-source Product Quantization
k-Nearest Neighbors implementation by Jeroen Van Der Donckt:

- Author: Jeroen Van Der Donckt
- Repository: https://github.com/jvdd/pqknn
- Reference revision:
  https://github.com/jvdd/pqknn/tree/4722d9be07bb12a9ba32e2fc9205dc54b8daf219
- Upstream `product_quantization/util.py`:
  https://github.com/jvdd/pqknn/blob/4722d9be07bb12a9ba32e2fc9205dc54b8daf219/product_quantization/util.py
- Upstream license:
  https://github.com/jvdd/pqknn/blob/4722d9be07bb12a9ba32e2fc9205dc54b8daf219/LICENSE
- License: MIT

Within `product_quantization/util.py`, the following functions are derived
from that upstream implementation:

- `log_nb_clusters_to_np_int_type`
- `squared_euclidean_dist`

These functions remain under the MIT License.

For clarity and to avoid a mixed license within the bodies of these two
functions, any later modifications made by Christian Jesinghaus specifically
within these two functions are also made available under the MIT License.

The applicable third-party MIT notice is reproduced in:

`THIRD_PARTY_LICENSE/classical_PQKNN_MIT.txt`

Other portions of the repository derived from the upstream PQKNN project
remain subject to the corresponding third-party MIT terms where applicable.

## 2. Quantum-related additions

The additional quantum-related helper implementations in
`product_quantization/util.py` were developed for the hybrid
quantum-classical Product Quantization project.

This includes, among others:

- `amplitude_encoding`
- `quantum_minimum_finding_oracle`
- `quantum_grover_minimum_search`
- `quantum_parallel_cluster_assignment`
- `quantum_distance_calculation_parallel`
- `quantum_swap_test_distance`
- `quantum_superposition_centroid_update`
- `quantum_superposition_mean`
- `quantum_amplitude_estimation_distance`

These functions were not intentionally adapted from a separate third-party
software implementation.

They were initially generated with the assistance of ChatGPT and were
subsequently reviewed by Christian Jesinghaus, tested in the context of the
project, integrated into the software, and in places manually modified.

The use of generative AI during development was also disclosed in the
Bachelor's thesis associated with the original project.


To the extent that Christian Jesinghaus holds copyright or other applicable
rights in the quantum-related additions, modifications, selection,
arrangement, integration, or other project-specific contributions, those
rights are licensed under the repository-level `LicenseRef-BA-Citation`
terms contained in `LICENSE`.

No separate third-party implementation or source-code version is identified
for the quantum helper functions because no specific external implementation
was intentionally used as their source.

## 3. Relationship between the licenses

The repository therefore contains separately licensed material.

The upstream-derived portions identified above remain available under the
MIT License.

Project-specific contributions for which Christian Jesinghaus holds
applicable rights are made available under `LicenseRef-BA-Citation`, as
described in the repository-level `LICENSE`.



## 4. Research citation

For the required academic citation of the project-specific contributions,
refer to the repository-level `LICENSE` and `CITATION.cff`.

The citation requirement does not alter the MIT terms applicable to
third-party code derived from Jeroen Van Der Donckt's `jvdd/pqknn`
implementation.
