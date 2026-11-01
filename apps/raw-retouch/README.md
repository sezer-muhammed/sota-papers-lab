# raw-retouch

Automatic RAW photo retouching: develop, search Lightroom-style sliders, judge with Claude, export JPEG and XMP

| | |
|---|---|
| Ops | `raw_retouch.scale` |
| Kernels | `cuda/kernels.cu` (NVRTC at runtime) |
| Paper | see `paper.toml` |

Run alone: `spl run raw_retouch.scale --input '{"n": 1048576}'`

## Exports, metadata and zoom

- Full-resolution files: JPEG (default), PNG (default, `export.png_bits` 8 or 16, lossless sRGB, deflated in parallel), 16-bit TIFF (`export.tiff16`) and the XMP sidecar.
- `RawInfo.captured_at` is ISO 8601 from EXIF DateTimeOriginal (with OffsetTimeOriginal when present); `gps` comes from EXIF GPS, and `place` names the nearest city offline (`src/place.rs`). Positions never leave the host.
- `raw_retouch.render` with `region` returns that rectangle of the upright full-resolution image at 1:1, exactly as the export has it. The first region of a photo and settings renders the whole image (or reuses the export's); later regions only crop it.

## Attribution

`geonames/cities.tsv.gz` is derived from [GeoNames](https://www.geonames.org) (cities15000, countryInfo and admin1CodesASCII), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It was reduced to name, country, first-level division and coordinates by `geonames/reduce.py`.
