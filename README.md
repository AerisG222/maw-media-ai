[![GitHub license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/AerisG222/maw-media-ai/blob/master/LICENSE.md)

# maw-media-ai

Tooling to use AI to automate scanning photos for objects / scenes / faces so that can be included in search.  Vibe coded
with Claude Sonnet 4.6

## Preparation

### Linux

Setup script (photo-tagger.sh) relies on conda.  You may need to run `sudo dnf install conda`, or your distro's equivalent,
if not already installed.

### Python Environment

`./photo-tagger.sh`

### Reference Photos for Facial Recognition

known_people/
    Alice/
        photo1.jpg
        photo2.jpg
    Bob/
        photo1.jpg

## Usage

# 1. Enroll known people (do this once)

`python photo_tagger.py enroll --known ./known_people --db faces.db`

# 2. Scan your photo library

`python photo_tagger.py scan --photos ./my_photos --db faces.db --output results.csv`

# 3. View a summary

`python photo_tagger.py report --output results.csv`


## License

MIT
