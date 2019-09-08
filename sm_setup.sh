#!/usr/bin/env sh
pip install -r requirements.txt
cd samples/cityscape
python cityscape.py $@