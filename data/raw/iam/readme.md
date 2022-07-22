# IAM Dataset

The IAM Handwriting Database contains forms of handwritten English text which can be used to train and test handwritten text recognizers and to perform writer identification and verification experiments.

- 657 writers contributed samples of their handwriting
- 1,539 pages of scanned text
- 13,353 isolated and labeled text lines

- http://www.fki.inf.unibe.ch/databases/iam-handwriting-database

## Pre-processing

First, all forms were placed into one directory called `forms`, from original directories like `formsA-D`.

To save space, I converted the original PNG files to JPG, and resized them to half-size
```
mkdir forms-resized
cd forms
ls -1 *.png | parallel --eta -j 6 convert '{}' -adaptive-resize 50% '../forms-resized/{.}.jpg'
```

## Split

The data split we will use is loosely based on the IAM Lines Large Writer Independent Text Line Recognition Task (`lwitlrt`) which provides 4 data splits:
 - Train: has 6,161 text lines from 747 pages written by 283 writers
 - Validation 1: has 900 text lines from 105 pages written by 46 writers
 - Validation 2: has 940 text lines from 115 pages written by 43 writers
 - Test: has 1,861 text lines from 232 pages written by 128 writers
Total: has 9,862 text lines from 1199 pages written by 500 writers
The text lines of all data sets are mutually exclusive, thus each writer has contributed to one set only.

The total text lines (9,862) in the data splits is way less then all the text lines (13,353) in the dataset. This is because:
 - pages of 157 writers (`657-500`) are not included in the data splits
 - 511 text lines are dropped from the 1,199 pages included in the data splits

To avoid missing out on all the dropped data, we slightly modify the data splits. We:
 - use all text lines in a page and never drop text lines
 - merge Validation 1 and Validation 2 into a single Validation data split
 - all the missing pages of 157 writers are added to the train data split.
Our final data splits are:
 - Train: has 9,462 text lines from 1,087 pages written by 440 writers
 - Validation: has 1,926 text lines from 220 pages written by 89 writers
 - Test: has 1,965 text lines from 232 pages written by 128 writers
Total: has 13,353 text lines from 1,539 pages written by 657 writers
