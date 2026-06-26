| Metric                    | Value                     | Notes                                           |
|:--------------------------|:--------------------------|:------------------------------------------------|
| Number of Specimens       | 8                         | All archosaurs, ESRF Paleo CT                   |
| Projection Images (total) | 240                       | 8 specimens × 30                                |
| Projections per Specimen  | 30                        | 6 ortho + 24 rotational                         |
| Orthographic Views        | 6 (3 axes × 2 directions) | Coronal, sagittal, transverse + flips           |
| Rotational Views          | 24 (0°–345°, Δ15°)        | 2D-after-MIP (sagittal MIP rotated)             |
| Rotation Step             | 15°                       | Covers full 360°                                |
| Synthetic Images (total)  | 960                       | 240 inputs × 4 deformations                     |
| Deformation Types         | 4                         | Compression, shearing, stretching, dissolution  |
| Images per Deformation    | 240                       | Balanced; seed=42                               |
| Binary Masks (total)      | 960                       | One mask per synthetic image                    |
| Image Resolution          | 224 × 224 px              | INTER_AREA resize from raw MIP                  |
| Image Format              | PNG (uint8 grayscale)     | Grayscale, no color channels                    |
| Random Seed               | 42                        | Reproducible with np.random.default_rng(42)     |
| Label Format              | CSV (11 columns)          | image_id, specimen, deformation_type, params, … |