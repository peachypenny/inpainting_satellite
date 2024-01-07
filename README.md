# Cloud Inpainting Satellite Imagery for LST

Several studies have worked on generating maps for urban heat islands (UHIs) using satellite imagery, but they focus on days with little cloud cover since satellite instruments can't "see" through clouds. This working repository documents our work in creating a U-Net model using Pytorch to generate maps for UHIs regardless of cloud coverage.

## Acknowledgements

### Github Repositories
| Name | Description |
| --- | --- |
| [Segmentation Models](https://github.com/qubvel/segmentation_models.pytorch) | Python library with neural networks for image segmentation (PyTorch) |
| [pylandtemp](https://github.com/pylandtemp/pylandtemp) | Simple API for computing global LST from NASA's Landsat images |
| [CU Boulder Final Proj.](https://github.com/paulaeperez/ea-uhi-final-project?tab=readme-ov-file) | Workflow to process and downscale ECOSTRESS LST data |
| [earthaccess](https://github.com/nsidc/earthaccess/) | Python library to search, download or stream NASA Earth science data |

### Papers Consulted
* [Machine learning enhanced gap filling in global land surface temperature analysis](https://blogs.reading.ac.uk/weather-and-climate-at-reading/2023/machine-learning-enhanced-gap-filling-in-global-land-surface-temperature-analysis/)
* [Analyzing Land Surface Temperature (LST) with Landsat 8 Data in Google Earth Engine](https://medium.com/@ridhomuh002/analyzing-land-surface-temperature-lst-with-landsat-8-data-in-google-earth-engine-f4dd7ca28e70)
