#!/bin/sh

echo "deb https://packages.cloud.google.com/apt gcsfuse-bionic main" > /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
apt -qq update
apt -qq install gcsfuse

# mount mimic dataset
mkdir ../data/mimic-cxr-data
gcsfuse --implicit-dirs mimic-cxr-data ../data/mimic-cxr-data

# mount chexpert dataset
mkdir ../data/chexpert-test
gcsfuse --implicit-dirs chexpert-test ../data/chexpert-test

# mount models
mkdir ../data/chexzero
gcsfuse --implicit-dirs chexzero ../data/chexzero

# mount experiments
mkdir ../data/chexzero-experiments
gcsfuse --implicit-dirs chexzero-experiments ../data/chexzero-experiments