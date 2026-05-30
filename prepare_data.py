"""
prepare_data.py
===============
Download the three public SocioPatterns contact datasets and convert each into
a weighted static edge list  `i j w`  where w = cumulative contact duration
(number of 20-second proximity intervals between i and j over the recording).

Outputs (read automatically by empirical_validation.py):
    data/primaryschool.csv     Stehle et al. 2011,  PLoS ONE 6(8):e23176
    data/highschool_2013.csv   Fournet & Barrat 2014, PLoS ONE 9:e107878 (Thiers13)
    data/hospital_ward.csv     Vanhems et al. 2013, PLoS ONE 8:e73970

The raw files are openly available from http://www.sociopatterns.org/datasets/
(CC-BY-NC-SA / CC0). This script tries SocioPatterns first, then public GitHub
mirrors. If both fail (e.g. no internet on the runner), drop the raw files in
data/raw/ manually and re-run; conversion is offline.
"""
import os, io, gzip, urllib.request
from collections import Counter
import networkx as nx

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "data")
RAW  = os.path.join(DATA, "raw")
os.makedirs(RAW, exist_ok=True)

# (filename in data/raw, list of candidate URLs)
SOURCES = {
    "tij_Thiers13.dat": [
        "https://raw.githubusercontent.com/YasXV/Social_network/main/tij_Thiers13.dat",
        "http://www.sociopatterns.org/wp-content/uploads/2015/07/High-School_data_2013.csv.gz",
    ],
    "detailed_list_of_contacts_Hospital.dat": [
        "https://raw.githubusercontent.com/FilippoGaravaglia/sociopatterns-epidemics/main/data/raw/detailed_list_of_contacts_Hospital.dat",
        "http://www.sociopatterns.org/wp-content/uploads/2013/09/detailed_list_of_contacts_Hospital.dat_.gz",
    ],
    # primary school comes as an aggregated graphml in this mirror:
    "sp_data_school_day_1.graphml": [
        "https://raw.githubusercontent.com/averma10/PS-Cumulative-Networks/main/Data/Source/sp_data_school_day_1.graphml",
    ],
}

def fetch(fname, urls):
    out = os.path.join(RAW, fname)
    if os.path.isfile(out):
        return out
    for url in urls:
        try:
            print(f"  downloading {fname} <- {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=60).read()
            if url.endswith(".gz"):
                data = gzip.decompress(data)
            with open(out, "wb") as f:
                f.write(data)
            return out
        except Exception as e:
            print(f"    failed: {e}")
    return None

def counts_from_temporal(path, i_col=1, j_col=2):
    c = Counter()
    with open(path) as f:
        for ln in f:
            p = ln.split()
            if len(p) > max(i_col, j_col) and p[i_col] != p[j_col]:
                c[tuple(sorted((p[i_col], p[j_col])))] += 1
    return c

def counts_from_graphml(path, attr="count"):
    g = nx.read_graphml(path); c = {}
    for u, v, d in g.edges(data=True):
        if u != v:
            c[tuple(sorted((u, v)))] = int(float(d.get(attr, d.get("weight", 1))))
    return c

def write_edges(counts, out):
    # relabel node ids to 0..n-1 for portability
    nodes = sorted({x for e in counts for x in e})
    idx = {n: i for i, n in enumerate(nodes)}
    with open(out, "w") as f:
        f.write("# i j weight(=cumulative 20s contact intervals)\n")
        for (a, b), w in counts.items():
            f.write(f"{idx[a]} {idx[b]} {int(w)}\n")
    print(f"  wrote {out}  ({len(counts)} edges, {len(nodes)} nodes)")

def main():
    print("Preparing empirical contact networks...")
    hs = fetch("tij_Thiers13.dat", SOURCES["tij_Thiers13.dat"])
    hw = fetch("detailed_list_of_contacts_Hospital.dat",
               SOURCES["detailed_list_of_contacts_Hospital.dat"])
    ps = fetch("sp_data_school_day_1.graphml",
               SOURCES["sp_data_school_day_1.graphml"])
    if hs: write_edges(counts_from_temporal(hs), os.path.join(DATA, "highschool_2013.csv"))
    if hw: write_edges(counts_from_temporal(hw), os.path.join(DATA, "hospital_ward.csv"))
    if ps: write_edges(counts_from_graphml(ps), os.path.join(DATA, "primaryschool.csv"))
    print("Done.")

if __name__ == "__main__":
    main()
