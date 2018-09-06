# LarschAndBaier2018

Code repository accompanying Larsch & Baier 2018, Biological motion as an innate perceptual mechanism driving social affiliation, Current Biology.

Python code for data analysis from raw data to final figures (Jupyter notebooks & python modules).

Bonsai workflows for virtual shoaling setup.

The Raw data is available from figshare: https://doi.org/10.6084/m9.figshare.6939923



# Data analysis

The analysis was written in Python 3.6
The entire analysis workflow from raw data to composite figures can be performed using the master jupyter notebook 2018_00_GenerateAllFigures.ipynb
Individual jupyter notebooks select data for specific figure panels, call analysis modules and run statistics and plotting functions.
Data loading, collection of meta data and computation of animal interactions are implemented in python modules that save intermediate files in the folder 'processing'.
41 Individual figure panels and composite figures are saved in the 'output' folder.

Size of the uncompressed raw Data: 9.2 GB
Size of intermediate processing files: 3 GB

Analysis was tested on several Windows 7 64bit and Windows 10 64bit PCs with 16 GB RAM.

## Quick start:

	1) Download the RawData.7z (~1.5 GB) and extract into an empty parent folder
	    https://doi.org/10.6084/m9.figshare.6939923
	
	2) Clone this bitbucket repository containing python code and jupyter notebooks:
		https://bitbucket.org/mpinbaierlab/larschandbaier2018

	
	3) In the local repository, locate props.csv in the sub folder 'LarschAndBaier2018"
		Adjust paths for "BaseDir" and "allExpFn" to your extracted RawData location
	
	4) Install Miniconda 64bit Python 3.6
		https://conda.io/miniconda.html
		This code was tested on Miniconda3-4.5.4-Windows-x86_64 
		https://repo.continuum.io/miniconda/
	
	5) Install the analysis python environment.yml, located in the local repository:
		in the anaconda prompt, type: conda env create --file=environment.yml
		This will install all the necessary python packages.
		
	6) start the jupyter notebook server:
		in the anaconda prompt, type: ipython notebook
		The notebook browser should open
		
	7) In the notebook browser, locate the folder LarschAndBaier2018
		Open the notebook 2018_00_GenerateAllFigures.ipynb
		Run all cells: From the top menu, select Cell>>Run All
		
	Analysis of all data takes 30-60 minutes
	
	->> The output figures can be found in a new sub folder 'output' where the RawData is located.