# Data Preprocessing

## Overview

Open Building Insights's objective is to produce high-quality building footprint datasets, by leveraging publicly available data, machine learning and geospatial analytics. The building footprints are obtained from Google-Microsoft Open Buildings (combined and published by [VIDA](https://source.coop/vida/google-microsoft-open-buildings)) and merged with select buildings from OpenStreet Maps. The building footprint catalog is further enriched by other data from public sources to provide fine grained information on the building level.

This section outlines the required data pre-processing steps in order to generate the building footprint dataset. The associated notebooks containing the code are listed in [Notebooks](#notebooks).


## VIDA data preprocessing 

The publicly available [Google-Microsoft Open Buildings (VIDA)](https://source.coop/vida/google-microsoft-open-buildings) contains building footprints that are available both globally and by country in various formats. The dataset contains a polygon defining the building's footprint on the ground, a confidence score indicating certainity of it being a building, and the respective source (either Google or Microsoft). 

In this workflow, the data was accessed in **GeoParquet** format using one of two methods: 
  1. **By country using s2 partitions (India)**: In this approach, all S2 partition files available for the target country are first identified from the VIDA country partition directory (published as GeoParquet partitions by S2 grid). Each partition filename is parsed into an ``S2 cell ID``, and the ID is converted into a geographic point by computing the centroid coordinates (lat/lon). The S2 points are then spatially filtered against a buffered area of interest (AOI) and the correspoding files downloaded. The selected S2 IDs are used to extract buildings from the downloaded partitions.
  
  2. **By country as a single file (Kenya)**: For Kenya, the data is downloaded as a single country-level GeoParquet file. This approach is used because the relevant S2 cells cover a much larger extent and their centroid coordinates fall outside the country boundaries.

  After download, the buildings are filtered using a minimum confidence threshold of **0.7**. In addition to ``geometry``, ``footprint source``, and ``confidence``, the following attributes were computed from each footprint polygon:

    - Area in meters
    - Perimeter in meters 
    - Number of building faces

## Building height calculation

Google’s [Open Buildings 2.5D Temporal Dataset](https://sites.research.google/gr/open-buildings/temporal/) is used as the source of building height rasters. Before executing the download script, the GeoTIFF tiles covering the AOI are identified using a [Colab Notebook](https://colab.research.google.com/github/google-research/google-research/blob/master/building_detection/open_buildings_temporal_download_region_geotiffs.ipynb) provided by Google, which generates a list of tile URLs for the AOI. This list is then used to download the relevant raster files.

Building heights are calculated by sampling raster pixels inside each building footprint polygon. For each GeoTIFF tile, the script extracts the height layer, reprojects the raster to CRS (Coordinate Reference System) matching the building footprints, and processes the tile in overlapping windows to keep memory use manageable. Within each window, buildings are preselected using centroid bounding boxes, then a polygon mask is applied so that only pixels inside the footprint contribute to the height estimate (no-data values are treated as missing).

For each building, the script computes three height statistics: ``height_mean``, ``height_median``, and ``height_max`` based on valid pixels. Two other attributes are derived from the height: ``floors`` (estimated number of storeys) and ``GFA`` (gross floor area), where GFA is calculated as ``footprint_area × floors``.

The follwing assumptions are applied: 
 - If a building’s height cannot be computed (i.e., the result is NaN), it is assigned a default height of 4.5 m. 
 - Heights are standardized into discrete values for consistency: 
    - any height up to 4.5 m is treated as a single-storey building and set to 4.5 m
    - heights between 4.5 m and 7.5 m are treated as two-storey buildings and set to 7.5 m
    - heights above 7.5 m are rounded up to the next 3 m step (for example, 8.5 m → 10.5 m, 11.0 m → 13.5 m, etc.).

## Urban / Rural classification

The [Global Human Settlement Layer – Settlement Model (GHS-SMOD)](https://human-settlement.emergency.copernicus.eu/download.php?ds=smod) is used as the source for urban/rural classification. GHSL-SMOD is distributed as a global raster where each pixel stores a settlement class code, provided at 1 km (1000 m) resolution. In this workflow, the raster is used to assign an urbanization label to each building footprint in the AOI.

The workflow follows two main steps. First, the global SMOD raster is reprojected to the same CRS as the building data and clipped to the AOI boundary. The clipped raster is then converted into polygons for each settlement class by extracting the areas covered by the relevant SMOD classes and writing them as GeoJSON features (one feature type per class). Second, building centroids are spatially matched to these polygons using a point-in-polygon check, and the corresponding class label is written to the building dataset. 

Two label sets `urban_split` and `ghsl_smod` are produced and added to the building dataset using two segregation styles:

* **Overview (`urban_split`)**: Classes used are - `Urban`, `Suburban`, `Rural`
* **Detailed (`ghsl_smod`)**: Classes used are - `Urban Center`, `Dense Urban`, `Semi-Dense Urban`, `Suburban / Peri-Urban`, `Rural Cluster`, `Low Density Rural`

Buildings that are not matched to any GHSL-SMOD class polygon during the spatial join are assigned `Rural` by default and, for the detailed segregation `Low Density Rural`.

## Elevation calculation

Elevation rasters are obtained from [EarthEnv-DEM90](https://www.earthenv.org/DEM.html) provided at ~90 m resolution. Before assigning elevations to buildings, the script identifies which DEM tiles are needed for the AOI by reading the boundary GeoJSON and extracting its bounding box (min/max longitude and latitude). Since EarthEnv-DEM90 is published as **5° × 5° tiles**, the AOI bounds are rounded to the nearest 5 degrees, and the corresponding tile URLs are generated (e.g., `EarthEnv-DEM90_NxxEyyy.tar.gz`). Each required tile archive is downloaded from the [EarthEnv mirror](https://datacommons.cyverse.org/browse/iplant/home/shared/earthenv_dem_data/EarthEnv-DEM90) and extracted, keeping only the files needed for processing (`.bil`, `.hdr`, `.prj`) in a local `elevation_rasters/` folder.

Elevations are then added to the building dataset. For each DEM tile, the script reads the raster metadata from the `.hdr` file (grid size, pixel size, upper-left origin, and nodata value), loads the `.bil` elevation array, and replaces nodata with `NaN`. To avoid processing unnecessary tiles, DEM tiles that overlap the building dataset extent are selected by checking tile bounding boxes derived from the `.hdr` metadata (using a coarse 0.2° sampling grid to speed up the overlap check). Buildings are then processed tile-by-tile by selecting only those whose centroids fall inside the tile bounds, converting each centroid coordinate to raster pixel indices, and reading the corresponding elevation value from the DEM grid. Intermediate parquet outputs are written per tile and finally merged into a single output file containing an added `elevation` column for each building.

## Implementation Details

The list of most important libraries is provided:

| Package Name | Short Description |
| --- | --- |
| geopandas | Geographic pandas extensions |
| pandas | Powerful data structures for data analysis, time series, and statistics |
| numpy | Fundamental package for array computing in Python |
| shapely | Manipulation and analysis of geometric objects |
| pyproj | Python interface to PROJ (cartographic projections and coordinate transformations library) |
| pyarrow | Columnar in-memory format and Parquet/Arrow I/O library |
| rasterio | Fast and direct raster I/O for use with NumPy and SciPy |
| rioxarray | geospatial xarray extension powered by rasterio |
| matplotlib | Python plotting package |
| plotly | Interactive graphing library for Python |
| requests | Python HTTP for Humans. |
| tqdm | Fast, extensible progress bar for Python and CLI |
| boto3 | AWS SDK for Python (used for S3-compatible object storage access) |
| botocore | Low-level, data-driven core of boto 3. |
| ibm_boto3 | The IBM SDK for Python |
| jaydebeapi | Use JDBC database drivers from Python 2/3 or Jython with a DB-API. |
| jpype | A Python to Java bridge. |
| s2cell | S2 cell ID utilities (e.g., converting S2 IDs to lat/lon) |


## Execution Details

The following notebooks are executed in order:

- 1_download_VIDA_S2grid_datasets.ipynb || 1_download_VIDA_datasets_Kenya.ipynb
-	2_filter_and_extract_buildings_from_VIDA_S2_Partitions.ipynb || 2_filter_and_extract_buildings_data-Kenya.ipynb
-	3_download_google_25D.ipynb
-	4_building_height_calculation_parquet.ipynb
-	5_rural_urban_json_segregation.ipynb
-	6_urban_rural_segregation_parquet_ver.ipynb
-	7_elevation_calculation.ipynb

