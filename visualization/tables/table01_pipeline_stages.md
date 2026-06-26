| Stage            | Responsible   | Input                   | Output                           | Key Tool                                            |
|:-----------------|:--------------|:------------------------|:---------------------------------|:----------------------------------------------------|
| Data Preparation | Person 1      | Raw ZIP archives        | Extracted folders                | File system                                         |
| Organization     | Person 1      | Organized folders       | Hierarchical dataset tree        | shutil.move                                         |
| Cleaning         | Person 1      | Unsorted JP2 slices     | Renamed sequential slices        | rename_sequential()                                 |
| Projection       | Person 2      | Sequential JP2 slices   | 30 PNG views/specimen            | stream_volume_projections()                         |
| Deformation      | Person 3      | 224×224 PNG projections | 960 synthetic images + 960 masks | apply_compression/shearing/stretching/dissolution() |