//! Offline reverse geocoding: a GPS position → the nearest city, its
//! province or state and its country ("Bornova, İzmir, Türkiye"). Owner:
//! media.
//!
//! Positions are private, so nothing goes to a geocoding service: the table
//! is bundled (`geonames/cities.tsv.gz`, about 0.5 MB): GeoNames'
//! cities15000 (every city of 15,000 people or more, about 34,000), with
//! country and first-level division names, reduced to name, country,
//! division, latitude and longitude. GeoNames (<https://www.geonames.org>),
//! CC BY 4.0. It is decompressed on first use and kept for the process.

use std::collections::HashMap;
use std::io::Read;
use std::sync::LazyLock;

use flate2::read::GzDecoder;

/// The bundled table: comment lines start with `#`; `c` lines name a
/// country (`c\tTR\tTürkiye`), `a` lines a first-level division
/// (`a\tTR.35\tİzmir Province`), and every other line is a city
/// (`İzmir\tTR\t35\t38.4127\t27.1384`).
static TABLE: &[u8] = include_bytes!("../geonames/cities.tsv.gz");

/// Farther than this from every city, a position has no place.
const MAX_KM: f64 = 300.0;
/// Farther than this from the nearest city, the place reads "near ...".
const NEAR_KM: f64 = 50.0;
const EARTH_KM: f64 = 6371.0;

/// Suffixes GeoNames puts on division names that read oddly in a place line.
const DIVISION_SUFFIXES: [&str; 5] = [
    " Province",
    " Region",
    " Governorate",
    " Prefecture",
    " Oblast",
];

struct City {
    name: String,
    country: String,
    division: String,
    /// Unit vector on the sphere: the nearest city has the largest dot product.
    at: [f64; 3],
}

struct Places {
    countries: HashMap<String, String>,
    divisions: HashMap<String, String>,
    cities: Vec<City>,
}

static PLACES: LazyLock<Places> = LazyLock::new(|| {
    let mut text = String::new();
    // The table is part of the binary; a failure here is a build defect,
    // which the tests catch. Degrade to "no place" rather than panic.
    if GzDecoder::new(TABLE).read_to_string(&mut text).is_err() {
        text.clear();
    }
    Places::parse(&text)
});

impl Places {
    fn parse(text: &str) -> Self {
        let mut places = Places {
            countries: HashMap::new(),
            divisions: HashMap::new(),
            cities: Vec::new(),
        };
        for line in text.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            match f.as_slice() {
                [comment, ..] if comment.starts_with('#') => {}
                ["c", code, name] => {
                    places.countries.insert((*code).into(), (*name).into());
                }
                ["a", code, name] => {
                    places.divisions.insert((*code).into(), (*name).into());
                }
                [name, country, division, lat, lon] => {
                    if let (Ok(lat), Ok(lon)) = (lat.parse(), lon.parse()) {
                        places.cities.push(City {
                            name: (*name).into(),
                            country: (*country).into(),
                            division: (*division).into(),
                            at: unit(lat, lon),
                        });
                    }
                }
                _ => {}
            }
        }
        places
    }

    fn nearest(&self, lat: f64, lon: f64) -> Option<(&City, f64)> {
        let p = unit(lat, lon);
        let (city, dot) = self
            .cities
            .iter()
            .map(|c| (c, c.at[0] * p[0] + c.at[1] * p[1] + c.at[2] * p[2]))
            .max_by(|a, b| a.1.total_cmp(&b.1))?;
        Some((city, EARTH_KM * dot.clamp(-1.0, 1.0).acos()))
    }

    fn name(&self, lat: f64, lon: f64) -> Option<String> {
        let (city, km) = self.nearest(lat, lon)?;
        if km > MAX_KM {
            return None;
        }
        let mut parts = vec![city.name.as_str()];
        if let Some(division) = self
            .divisions
            .get(&format!("{}.{}", city.country, city.division))
        {
            let division = DIVISION_SUFFIXES
                .iter()
                .find_map(|s| division.strip_suffix(s))
                .unwrap_or(division);
            if !division.is_empty() && division != city.name {
                parts.push(division);
            }
        }
        parts.push(
            self.countries
                .get(&city.country)
                .map_or(city.country.as_str(), String::as_str),
        );
        let place = parts.join(", ");
        Some(if km > NEAR_KM {
            format!("near {place}")
        } else {
            place
        })
    }
}

fn unit(lat: f64, lon: f64) -> [f64; 3] {
    let (lat, lon) = (lat.to_radians(), lon.to_radians());
    [lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()]
}

/// The nearest city to a position, with its division and country, or
/// `None` far from any city (at sea) or for an invalid position.
/// Decompresses the table on first use (tens of milliseconds); a lookup
/// scans it (well under a millisecond).
pub fn place_name(lat: f64, lon: f64) -> Option<String> {
    let valid = lat.is_finite() && lon.is_finite() && lat.abs() <= 90.0 && lon.abs() <= 180.0;
    if !valid {
        return None;
    }
    PLACES.name(lat, lon)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_bundled_table_loads() {
        let places = &*PLACES;
        assert!(places.cities.len() > 30_000, "{}", places.cities.len());
        assert!(places.countries.len() > 200);
        assert_eq!(places.countries["TR"], "Türkiye");
    }

    #[test]
    fn positions_name_their_nearest_city() {
        let cases = [
            // İzmir's Konak square; Bornova is a district of İzmir.
            ((38.4189, 27.1287), "İzmir, Türkiye"),
            ((38.4622, 27.2164), "Bornova, İzmir, Türkiye"),
            ((41.0082, 28.9784), "Istanbul, Türkiye"),
            ((48.8584, 2.2945), "Paris 16 Passy, Île-de-France, France"),
            ((35.6586, 139.7454), "Minato City, Tokyo, Japan"),
            ((-33.8568, 151.2153), "Sydney, New South Wales, Australia"),
        ];
        let got: Vec<_> = cases
            .iter()
            .map(|&((lat, lon), _)| place_name(lat, lon).unwrap_or_default())
            .collect();
        let want: Vec<_> = cases.iter().map(|&(_, want)| want).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn remote_and_invalid_positions_have_no_place() {
        assert_eq!(place_name(0.0, -30.0), None, "mid-Atlantic");
        assert_eq!(place_name(f64::NAN, 1.0), None);
        assert_eq!(place_name(91.0, 0.0), None);
        // 60 to 300 km from the nearest city.
        let far = place_name(-24.0, 133.0).unwrap_or_default();
        assert!(far.starts_with("near "), "{far}");
    }

    #[test]
    fn parse_skips_bad_lines() {
        let places = Places::parse(
            "# comment\nc\tXX\tExample\na\tXX.01\tNorth Region\nTown\tXX\t01\t10.0\t20.0\n\
             Broken\tXX\t01\tnot-a-number\t20\njunk\n",
        );
        assert_eq!(places.cities.len(), 1);
        assert_eq!(
            places.name(10.01, 20.0).as_deref(),
            Some("Town, North, Example")
        );
    }
}
