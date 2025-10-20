"""Code to parse the OlmoEarth Pretrain dataset.

parse.py takes care of parsing the CSVs in the raw dataset to identify all of the tiles
at which various modalities are available.

sample.py synthesizes this information across modalities to determine what is needed
for loading individual training samples.
"""
