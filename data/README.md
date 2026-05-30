# Empirical contact-network data

The empirical validation in Section 4.5 of the paper uses three real
high-resolution contact networks from the **SocioPatterns** initiative
(http://www.sociopatterns.org/datasets/).

These datasets are **not redistributed in this repository.** They are the
property of SocioPatterns and are released under their own terms
(CC-BY-NC-SA / CC0). The analysis code in this repository is MIT-licensed,
but that licence does **not** apply to the contact data.

## How to obtain the data

Run the download/conversion script from the repository root:

```bash
python prepare_data.py
```

This downloads the raw SocioPatterns records (with public GitHub mirrors as a
fallback) and writes three weighted static edge lists into this folder:

| File                   | Dataset                | Source |
|------------------------|------------------------|--------|
| `primaryschool.csv`    | Primary school         | Stehlé et al. 2011, *PLoS ONE* 6(8):e23176 |
| `highschool_2013.csv`  | High school (Thiers13) | Fournet & Barrat 2014, *PLoS ONE* 9:e107878 |
| `hospital_ward.csv`    | Hospital ward          | Vanhems et al. 2013, *PLoS ONE* 8:e73970 |

Each edge list has the form `i j w`, where `w` is the cumulative contact
duration (number of 20-second proximity intervals between nodes `i` and `j`).

If automatic download fails (e.g. no internet access), download the raw files
manually from SocioPatterns, place them in `data/raw/`, and re-run
`prepare_data.py`; the conversion step works offline.

`analyses/empirical_validation.py` reads these files automatically once they
are present.

Please cite the original dataset papers above when using these networks.
