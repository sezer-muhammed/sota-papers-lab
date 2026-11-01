//! Lightroom / Camera Raw XMP sidecar with every develop setting. Owner: judge.
//!
//! The layout copies what Lightroom Classic itself writes next to a RAW (no
//! packet wrapper, one attribute per line, signed integers such as "+15",
//! exposure as "+0.35", sharpening radius as "+1.0"), so Lightroom, Camera
//! Raw and ExifTool read it as their own. Every [`Params`] field goes under
//! its own `crs:` name from [`PARAM_SPECS`], with the parametric tone
//! curve's fixed split points ([`TONE_CURVE_SPLITS`]) and, when lens profile
//! corrections are on, `LensProfileSetup="LensDefaults"`, so Lightroom picks
//! the lens's profile itself. White balance is always written as Custom with
//! explicit Temperature and Tint, and HDR editing is switched off so a
//! user's HDR default cannot change the look.

use crate::types::{PARAM_SPECS, ParamSpec, Params, RawInfo, TONE_CURVE_SPLITS};

/// The Camera Raw version the settings claim to come from: the first one
/// that knows process version 15.4, so no Camera Raw that can apply them
/// treats them as written by a newer version.
const CRS_VERSION: &str = "15.4";
const PROCESS_VERSION: &str = "15.4";

/// A complete `.xmp` file; name it like the RAW (`DSC01234.xmp`) next to it.
pub fn sidecar(params: &Params, info: &RawInfo) -> String {
    let p = params.quantized();
    let mut fields: Vec<(String, String)> = [
        ("tiff:Make", escape(&info.make)),
        ("tiff:Model", escape(&info.model)),
        ("crs:Version", CRS_VERSION.to_owned()),
        ("crs:ProcessVersion", PROCESS_VERSION.to_owned()),
        ("crs:WhiteBalance", "Custom".to_owned()),
    ]
    .map(|(name, value)| (name.to_owned(), value))
    .into();
    for (spec, value) in PARAM_SPECS.iter().zip(p.to_array()) {
        fields.push((format!("crs:{}", spec.crs), value_of(spec, value)));
        match spec.key {
            // The split points follow the last region slider.
            "curve_shadows" => fields.extend(
                TONE_CURVE_SPLITS.map(|(name, split)| (format!("crs:{name}"), unsigned(split))),
            ),
            "lens_profile" if value >= 0.5 => {
                fields.push(("crs:LensProfileSetup".into(), "LensDefaults".into()));
            }
            _ => {}
        }
    }
    fields.extend(
        [
            ("crs:HasSettings", "True"),
            ("crs:HDREditMode", "0"),
            ("crs:AlreadyApplied", "False"),
        ]
        .map(|(name, value)| (name.to_owned(), value.to_owned())),
    );
    let mut out = String::with_capacity(4096);
    out.push_str(concat!(
        "<x:xmpmeta xmlns:x=\"adobe:ns:meta/\" x:xmptk=\"SOTA Papers Lab raw-retouch\">\n",
        " <rdf:RDF xmlns:rdf=\"http://www.w3.org/1999/02/22-rdf-syntax-ns#\">\n",
        "  <rdf:Description rdf:about=\"\"\n",
        "    xmlns:tiff=\"http://ns.adobe.com/tiff/1.0/\"\n",
        "    xmlns:crs=\"http://ns.adobe.com/camera-raw-settings/1.0/\"",
    ));
    for (name, value) in fields {
        out.push_str("\n   ");
        out.push_str(&name);
        out.push_str("=\"");
        out.push_str(&value);
        out.push('"');
    }
    out.push_str("/>\n </rdf:RDF>\n</x:xmpmeta>\n");
    out
}

/// One setting as Lightroom writes it: exposure in hundredths of a stop
/// ("+0.35"), sharpening radius in tenths of a pixel ("+1.0"), sliders
/// that can go negative with a sign ("+15", "-20", "0"), and the rest
/// (temperature, Detail amounts, the Lens Corrections switches as 1 or 0)
/// as plain integers.
fn value_of(spec: &ParamSpec, v: f32) -> String {
    match spec.key {
        "exposure" => exposure(v),
        "sharpen_radius" => tenths(v),
        _ if spec.min < 0.0 => signed(v),
        _ => unsigned(v),
    }
}

/// A whole non-negative value: "40", "5450".
fn unsigned(v: f32) -> String {
    (v.round().max(0.0) as i64).to_string()
}

/// One decimal with a sign, as Lightroom writes the sharpening radius: "+1.0".
fn tenths(v: f32) -> String {
    format!("{:+.1}", (v * 10.0).round() / 10.0)
}

/// A whole slider value as Lightroom writes it: "+15", "-20", "0".
pub(crate) fn signed(v: f32) -> String {
    let v = v.round() as i32;
    if v > 0 {
        format!("+{v}")
    } else {
        v.to_string()
    }
}

/// Exposure in EV with two decimals: "+0.35", "-1.20", "0.00".
pub(crate) fn exposure(v: f32) -> String {
    let hundredths = (v * 100.0).round() as i32;
    if hundredths == 0 {
        "0.00".to_owned()
    } else {
        format!("{:+.2}", hundredths as f32 / 100.0)
    }
}

/// Text safe inside a double-quoted XML attribute. Control characters
/// (trailing NULs in EXIF strings, for instance) are not legal XML and go.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s
        .trim_matches(|c: char| c.is_whitespace() || c == '\0')
        .chars()
    {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn info(make: &str, model: &str) -> RawInfo {
        RawInfo {
            make: make.into(),
            model: model.into(),
            lens: Some("FE 16-35mm F2.8 GM".into()),
            width: 6000,
            height: 4000,
            orientation: 1,
            iso: Some(400),
            exposure_time_s: Some(1.0 / 250.0),
            f_number: Some(4.0),
            focal_length_mm: Some(35.0),
            captured_at: None,
            gps: None,
            place: None,
            as_shot_temperature: 5200.0,
            as_shot_tint: 3.0,
            raw_clipped_fraction: 0.0,
        }
    }

    fn params() -> Params {
        Params {
            temperature: 5449.6,
            tint: 7.6,
            exposure: 0.346,
            contrast: 15.2,
            highlights: -40.4,
            shadows: 29.5,
            whites: 10.0,
            blacks: -5.0,
            vibrance: 20.0,
            saturation: -0.3,
            ..Params::neutral(5200.0, 0.0)
        }
    }

    /// [`params`] with a value away from Lightroom's default in every panel.
    fn every_panel() -> Params {
        Params {
            curve_highlights: -12.0,
            curve_lights: 8.4,
            curve_darks: -6.0,
            curve_shadows: 3.0,
            texture: 6.0,
            clarity: 11.6,
            dehaze: 4.0,
            hue_orange: -3.0,
            saturation_green: -9.0,
            saturation_blue: -14.0,
            luminance_orange: 7.0,
            luminance_blue: -18.0,
            shadow_tint: 2.0,
            red_primary_saturation: 5.0,
            blue_primary_hue: -4.0,
            sharpening: 46.0,
            sharpen_radius: 0.94,
            sharpen_detail: 22.0,
            sharpen_masking: 12.0,
            luminance_nr: 16.0,
            color_nr: 30.0,
            lens_profile: 1.0,
            lens_ca: 1.0,
            ..params()
        }
    }

    #[test]
    fn sidecar_matches_lightroom_layout() {
        let xmp = sidecar(&every_panel(), &info("SONY", "ILCE-7M3"));
        let want = r#"<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="SOTA Papers Lab raw-retouch">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"
   tiff:Make="SONY"
   tiff:Model="ILCE-7M3"
   crs:Version="15.4"
   crs:ProcessVersion="15.4"
   crs:WhiteBalance="Custom"
   crs:Temperature="5450"
   crs:Tint="+8"
   crs:Exposure2012="+0.35"
   crs:Contrast2012="+15"
   crs:Highlights2012="-40"
   crs:Shadows2012="+30"
   crs:Whites2012="+10"
   crs:Blacks2012="-5"
   crs:Vibrance="+20"
   crs:Saturation="0"
   crs:ParametricHighlights="-12"
   crs:ParametricLights="+8"
   crs:ParametricDarks="-6"
   crs:ParametricShadows="+3"
   crs:ParametricShadowSplit="25"
   crs:ParametricMidtoneSplit="50"
   crs:ParametricHighlightSplit="75"
   crs:Texture="+6"
   crs:Clarity2012="+12"
   crs:Dehaze="+4"
   crs:HueAdjustmentRed="0"
   crs:HueAdjustmentOrange="-3"
   crs:HueAdjustmentYellow="0"
   crs:HueAdjustmentGreen="0"
   crs:HueAdjustmentAqua="0"
   crs:HueAdjustmentBlue="0"
   crs:HueAdjustmentPurple="0"
   crs:HueAdjustmentMagenta="0"
   crs:SaturationAdjustmentRed="0"
   crs:SaturationAdjustmentOrange="0"
   crs:SaturationAdjustmentYellow="0"
   crs:SaturationAdjustmentGreen="-9"
   crs:SaturationAdjustmentAqua="0"
   crs:SaturationAdjustmentBlue="-14"
   crs:SaturationAdjustmentPurple="0"
   crs:SaturationAdjustmentMagenta="0"
   crs:LuminanceAdjustmentRed="0"
   crs:LuminanceAdjustmentOrange="+7"
   crs:LuminanceAdjustmentYellow="0"
   crs:LuminanceAdjustmentGreen="0"
   crs:LuminanceAdjustmentAqua="0"
   crs:LuminanceAdjustmentBlue="-18"
   crs:LuminanceAdjustmentPurple="0"
   crs:LuminanceAdjustmentMagenta="0"
   crs:ShadowTint="+2"
   crs:RedHue="0"
   crs:RedSaturation="+5"
   crs:GreenHue="0"
   crs:GreenSaturation="0"
   crs:BlueHue="-4"
   crs:BlueSaturation="0"
   crs:Sharpness="46"
   crs:SharpenRadius="+0.9"
   crs:SharpenDetail="22"
   crs:SharpenEdgeMasking="12"
   crs:LuminanceSmoothing="16"
   crs:ColorNoiseReduction="30"
   crs:LensProfileEnable="1"
   crs:LensProfileSetup="LensDefaults"
   crs:AutoLateralCA="1"
   crs:HasSettings="True"
   crs:HDREditMode="0"
   crs:AlreadyApplied="False"/>
 </rdf:RDF>
</x:xmpmeta>
"#;
        assert_eq!(xmp, want);
        check_well_formed(&xmp).unwrap();
    }

    /// The settings back from a sidecar's attributes, the way Lightroom
    /// reads them: every [`PARAM_SPECS`] name must be there.
    fn read_back(attrs: &HashMap<String, String>) -> Params {
        let mut a = [0.0; Params::COUNT];
        for (v, spec) in a.iter_mut().zip(&PARAM_SPECS) {
            let text = &attrs[&format!("crs:{}", spec.crs)];
            *v = text
                .parse()
                .unwrap_or_else(|_| panic!("crs:{} = {text:?}", spec.crs));
        }
        Params::from_array(a)
    }

    #[test]
    fn every_setting_round_trips() {
        let lens_off = Params {
            lens_profile: 0.0,
            lens_ca: 0.0,
            ..every_panel()
        };
        let extremes = Params {
            hue_red: -100.0,
            luminance_magenta: 100.0,
            sharpening: 150.0,
            sharpen_radius: 3.0,
            ..every_panel()
        };
        for p in [
            every_panel(),
            lens_off,
            extremes,
            params(),
            Params::neutral(5200.0, 0.0),
        ] {
            let xmp = sidecar(&p, &info("SONY", "ILCE-7M3"));
            let attrs = check_well_formed(&xmp).unwrap();
            assert_eq!(read_back(&attrs), p.quantized());
            // Nothing else under crs: but the fixed fields, the split points
            // and the lens profile setup.
            let fixed = [
                "Version",
                "ProcessVersion",
                "WhiteBalance",
                "HasSettings",
                "HDREditMode",
                "AlreadyApplied",
                "LensProfileSetup",
            ];
            for name in attrs.keys().filter_map(|k| k.strip_prefix("crs:")) {
                assert!(
                    PARAM_SPECS.iter().any(|s| s.crs == name)
                        || TONE_CURVE_SPLITS.iter().any(|(n, _)| *n == name)
                        || fixed.contains(&name),
                    "unexpected crs:{name}"
                );
            }
            for (name, split) in TONE_CURVE_SPLITS {
                assert_eq!(attrs[&format!("crs:{name}")].parse::<f32>(), Ok(split));
            }
            assert_eq!(
                attrs.get("crs:LensProfileSetup").map(String::as_str),
                (p.lens_profile >= 0.5).then_some("LensDefaults")
            );
            assert_eq!(attrs["crs:WhiteBalance"], "Custom");
        }
    }

    #[test]
    fn values_are_clamped_rounded_and_signed_like_lightroom() {
        let wild = Params {
            temperature: 90000.0,
            tint: -400.0,
            exposure: -0.004,
            contrast: 250.0,
            highlights: -0.4,
            shadows: 0.5,
            whites: -100.0,
            blacks: f32::from_bits(0x8000_0000), // -0.0
            vibrance: 99.5,
            saturation: -99.5,
            curve_lights: 140.0,
            hue_aqua: -0.4,
            luminance_blue: -101.0,
            blue_primary_saturation: 0.5,
            sharpening: 170.4,
            sharpen_radius: 0.27,
            sharpen_detail: -3.0,
            color_nr: 24.5,
            lens_profile: 0.7,
            lens_ca: 0.2,
            ..Params::neutral(5200.0, 0.0)
        };
        let xmp = sidecar(&wild, &info("SONY", "ILCE-7M3"));
        let attrs = check_well_formed(&xmp).unwrap();
        assert_eq!(attrs["crs:ParametricLights"], "+100");
        assert_eq!(attrs["crs:HueAdjustmentAqua"], "0");
        assert_eq!(attrs["crs:LuminanceAdjustmentBlue"], "-100");
        assert_eq!(attrs["crs:BlueSaturation"], "+1");
        assert_eq!(attrs["crs:Sharpness"], "150");
        assert_eq!(attrs["crs:SharpenRadius"], "+0.5");
        assert_eq!(attrs["crs:SharpenDetail"], "0");
        assert_eq!(attrs["crs:ColorNoiseReduction"], "25");
        assert_eq!(attrs["crs:LensProfileEnable"], "1");
        assert_eq!(attrs["crs:LensProfileSetup"], "LensDefaults");
        assert_eq!(attrs["crs:AutoLateralCA"], "0");
        assert_eq!(tenths(1.0), "+1.0");
        assert_eq!(tenths(2.96), "+3.0");
        assert_eq!(unsigned(39.5), "40");
        assert_eq!(attrs["crs:Temperature"], "50000");
        assert_eq!(attrs["crs:Tint"], "-150");
        assert_eq!(attrs["crs:Exposure2012"], "0.00");
        assert_eq!(attrs["crs:Contrast2012"], "+100");
        assert_eq!(attrs["crs:Highlights2012"], "0");
        assert_eq!(attrs["crs:Shadows2012"], "+1");
        assert_eq!(attrs["crs:Whites2012"], "-100");
        assert_eq!(attrs["crs:Blacks2012"], "0");
        assert_eq!(attrs["crs:Vibrance"], "+100");
        assert_eq!(attrs["crs:Saturation"], "-100");

        assert_eq!(exposure(-1.2), "-1.20");
        assert_eq!(exposure(5.0), "+5.00");
        assert_eq!(exposure(0.0151), "+0.02");
        let low = Params {
            temperature: 1500.0,
            ..params()
        };
        let attrs = check_well_formed(&sidecar(&low, &info("SONY", "X"))).unwrap();
        assert_eq!(attrs["crs:Temperature"], "2000");
    }

    #[test]
    fn camera_names_are_escaped() {
        let xmp = sidecar(&params(), &info("A&B \"Co\" <x>\0\0", "M'1\u{7}"));
        let attrs = check_well_formed(&xmp).unwrap();
        assert_eq!(attrs["tiff:Make"], "A&amp;B &quot;Co&quot; &lt;x&gt;");
        assert_eq!(attrs["tiff:Model"], "M&apos;1");
    }

    /// A strict little XML checker for what sidecars contain: elements,
    /// attributes and whitespace. Checks nesting, quoting, duplicate
    /// attributes, entity syntax and that every prefix is declared in scope.
    /// Returns the rdf:Description attributes (raw, unescaped text).
    fn check_well_formed(xml: &str) -> Result<HashMap<String, String>, String> {
        let b = xml.as_bytes();
        let mut i = 0;
        let mut stack: Vec<(String, Vec<String>)> = Vec::new();
        let mut description = HashMap::new();
        let mut seen_root = false;
        let skip_ws = |i: &mut usize| {
            while *i < b.len() && b[*i].is_ascii_whitespace() {
                *i += 1;
            }
        };
        let name = |i: &mut usize| -> Result<String, String> {
            let start = *i;
            while *i < b.len() && (b[*i].is_ascii_alphanumeric() || b":_-.".contains(&b[*i])) {
                *i += 1;
            }
            if start == *i || b[start].is_ascii_digit() {
                return Err(format!("bad name at {start}"));
            }
            Ok(xml[start..*i].to_owned())
        };
        loop {
            skip_ws(&mut i);
            if i == b.len() {
                break;
            }
            if b[i] != b'<' {
                return Err(format!("text outside elements at {i}"));
            }
            i += 1;
            if b.get(i) == Some(&b'/') {
                i += 1;
                let n = name(&mut i)?;
                skip_ws(&mut i);
                if b.get(i) != Some(&b'>') {
                    return Err(format!("bad end tag {n}"));
                }
                i += 1;
                match stack.pop() {
                    Some((open, _)) if open == n => {}
                    other => return Err(format!("</{n}> closes {other:?}")),
                }
                continue;
            }
            if stack.is_empty() && seen_root {
                return Err("two root elements".into());
            }
            seen_root = true;
            let tag = name(&mut i)?;
            let mut attrs: Vec<(String, String)> = Vec::new();
            let self_closing = loop {
                let had_ws = i < b.len() && b[i].is_ascii_whitespace();
                skip_ws(&mut i);
                match b.get(i) {
                    Some(b'/') if b.get(i + 1) == Some(&b'>') => {
                        i += 2;
                        break true;
                    }
                    Some(b'>') => {
                        i += 1;
                        break false;
                    }
                    Some(_) if had_ws => {
                        let n = name(&mut i)?;
                        if b.get(i) != Some(&b'=') || b.get(i + 1) != Some(&b'"') {
                            return Err(format!("attribute {n} not name=\"value\""));
                        }
                        i += 2;
                        let start = i;
                        while i < b.len() && b[i] != b'"' {
                            if b[i] == b'<' || (b[i] < 0x20 && b[i] != b'\t') {
                                return Err(format!("illegal character in {n}"));
                            }
                            if b[i] == b'&' {
                                let end = xml[i..].find(';').ok_or("unterminated entity")?;
                                let ent = &xml[i + 1..i + end];
                                if !["amp", "lt", "gt", "quot", "apos"].contains(&ent) {
                                    return Err(format!("unknown entity &{ent};"));
                                }
                            }
                            i += 1;
                        }
                        if i == b.len() {
                            return Err(format!("unterminated value of {n}"));
                        }
                        if attrs.iter().any(|(a, _)| *a == n) {
                            return Err(format!("duplicate attribute {n}"));
                        }
                        attrs.push((n, xml[start..i].to_owned()));
                        i += 1;
                    }
                    _ => return Err(format!("bad tag {tag}")),
                }
            };
            let declared: Vec<String> = attrs
                .iter()
                .filter_map(|(n, _)| n.strip_prefix("xmlns:").map(str::to_owned))
                .collect();
            let in_scope = |prefix: &str| {
                prefix == "xmlns"
                    || declared.iter().any(|d| d == prefix)
                    || stack.iter().any(|(_, ds)| ds.iter().any(|d| d == prefix))
            };
            for n in std::iter::once(&tag).chain(attrs.iter().map(|(n, _)| n)) {
                if let Some((prefix, _)) = n.split_once(':')
                    && !in_scope(prefix)
                {
                    return Err(format!("undeclared prefix in {n}"));
                }
            }
            if tag == "rdf:Description" {
                description = attrs.into_iter().collect();
            }
            if !self_closing {
                stack.push((tag, declared));
            }
        }
        if !stack.is_empty() {
            return Err(format!("unclosed {stack:?}"));
        }
        if !seen_root {
            return Err("empty document".into());
        }
        Ok(description)
    }

    #[test]
    fn checker_rejects_broken_xml() {
        for bad in [
            "<a><b></a></b>",
            "<a x=\"1\" x=\"2\"/>",
            "<a x='1'/>",
            "<p:a/>",
            "<a x=\"&bogus;\"/>",
            "<a/><b/>",
            "<a>",
            "",
        ] {
            assert!(check_well_formed(bad).is_err(), "{bad}");
        }
        assert!(check_well_formed("<p:a xmlns:p=\"u\"><p:b p:c=\"&amp;\"/></p:a>").is_ok());
    }
}
