"""Reduce GeoNames cities15000 + countryInfo + admin1 codes to a compact TSV.

Lines: "c<TAB>CC<TAB>country name", "a<TAB>CC.ADMIN1<TAB>admin1 name",
"<name><TAB>CC<TAB>ADMIN1<TAB>lat<TAB>lon" (degrees, 4 decimals).
"""
import os
import sys

src, out = sys.argv[1], sys.argv[2]
OVERRIDES = {"TR": "Türkiye"}
countries = {}
for line in open(os.path.join(src, "countryInfo.txt"), encoding="utf-8"):
    if line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if len(f) > 4:
        countries[f[0]] = OVERRIDES.get(f[0], f[4])
admin1 = {}
for line in open(os.path.join(src, "admin1CodesASCII.txt"), encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    if len(f) >= 2:
        admin1[f[0]] = f[1]
cities = []
used_cc, used_a1 = set(), set()
for line in open(os.path.join(src, "cities15000.txt"), encoding="utf-8"):
    f = line.rstrip("\n").split("\t")
    name, lat, lon, cc, a1 = f[1].strip(), float(f[4]), float(f[5]), f[8], f[10]
    # Sections of cities (PPLX) and historical, abandoned or destroyed
    # places name a neighbourhood or nothing; the city around them reads better.
    if not name or f[7] in ("PPLX", "PPLH", "PPLQ", "PPLW"):
        continue
    cities.append((name, cc, a1, lat, lon))
    used_cc.add(cc)
    if f"{cc}.{a1}" in admin1:
        used_a1.add(f"{cc}.{a1}")
with open(out, "w", encoding="utf-8") as o:
    o.write(
        "# GeoNames (https://www.geonames.org), CC BY 4.0: cities15000 without PPLX/PPLH/PPLQ/PPLW, "
        "countryInfo and admin1CodesASCII, reduced to name, country, admin1, lat, lon\n"
    )
    for cc in sorted(used_cc):
        if cc in countries:
            o.write(f"c\t{cc}\t{countries[cc]}\n")
    for k in sorted(used_a1):
        o.write(f"a\t{k}\t{admin1[k]}\n")
    for name, cc, a1, lat, lon in cities:
        o.write(f"{name}\t{cc}\t{a1}\t{lat:.4f}\t{lon:.4f}\n")
print(len(cities), "cities", len(used_cc), "countries", len(used_a1), "admin1")
