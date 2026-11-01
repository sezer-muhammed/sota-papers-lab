//! The photographer's edit request without Claude: keywords in Turkish and
//! English pick the looks to search and the one to choose, and the output
//! says honestly that only keywords were read.

use crate::objective::{LookSpec, look_offset, offset_params};
use crate::types::Params;

/// A style a request can name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Style {
    BlackWhite,
    HighContrast,
    Vivid,
    Natural,
    Moody,
    Bright,
    Skin,
}

impl Style {
    const ALL: [Style; 7] = [
        Style::BlackWhite,
        Style::HighContrast,
        Style::Vivid,
        Style::Natural,
        Style::Moody,
        Style::Bright,
        Style::Skin,
    ];

    pub(super) fn label(self) -> &'static str {
        match self {
            Style::BlackWhite => "Black and white",
            Style::HighContrast => "High contrast",
            Style::Vivid => "Vivid",
            Style::Natural => "Natural",
            Style::Moody => "Moody",
            Style::Bright => "Bright and airy",
            Style::Skin => "Warm skin",
        }
    }

    /// Words and phrases that name the style, as [`words`] folds them: a
    /// phrase matches consecutive words, and a trailing `*` matches any
    /// ending (Turkish suffixes: "kontrastlı", "doygunluk").
    fn keywords(self) -> &'static [&'static str] {
        match self {
            Style::BlackWhite => &[
                "siyah beyaz",
                "siyah ve beyaz",
                "siyahbeyaz",
                "black and white",
                "black white",
                "blackandwhite",
                "b and w",
                "b w",
                "bw",
                "monochrom*",
                "mono",
                "monokrom*",
                "greyscale",
                "grayscale",
                "gri ton*",
                "tek renk*",
                "noir",
            ],
            Style::HighContrast => &[
                "high contrast*",
                "higher contrast",
                "more contrast",
                "strong contrast",
                "contrasty",
                "punchy",
                "yuksek kontrast*",
                "kontrastli*",
                "guclu kontrast*",
                "sert kontrast*",
                "bol kontrast*",
                "daha kontrast*",
                "kontrasti yuksek",
                "kontrasti artir*",
            ],
            Style::Vivid => &[
                "canli*",
                "doygun*",
                "renkli",
                "saturated",
                "saturate",
                "high saturation",
                "vivid",
                "vibrant",
                "colorful",
                "colourful",
                "rich colo*",
            ],
            Style::Natural => &[
                "dogal*",
                "natural*",
                "realistic",
                "gercekci",
                "true to life",
                "optimum",
                "optimal",
                "balanced",
                "dengeli",
            ],
            Style::Moody => &[
                "sinematik",
                "sinema*",
                "cinematic",
                "cinema",
                "moody",
                "filmic",
                "film look",
                "movie",
                "motion*",
                "dramatic",
                "dramatik",
                "karanlik",
                "dark",
                "atmospheric",
                "atmosferik",
            ],
            Style::Bright => &[
                "bright*", "airy", "aydinlik", "ferah", "parlak", "high key", "pastel*",
            ],
            Style::Skin => &[
                "ten",
                "teni",
                "tenin",
                "ten rengi",
                "ten ton*",
                "cilt*",
                "skin*",
                "portre*",
                "portrait*",
                "face*",
            ],
        }
    }

    /// The look that stands for the style: a built-in look, or auto moved
    /// by the style's typical edit.
    fn look(self, auto: &Params) -> LookSpec {
        match self {
            Style::BlackWhite => LookSpec::black_and_white(),
            Style::Natural => LookSpec::natural(),
            Style::Vivid => builtin("vivid"),
            Style::Moody => builtin("moody"),
            Style::Bright => builtin("bright"),
            Style::HighContrast => {
                LookSpec::proposal("contrast", self.label(), moved(auto, &self.offset()))
            }
            Style::Skin => LookSpec::proposal("skin", self.label(), moved(auto, &self.offset())),
        }
    }

    /// The style's typical edit relative to auto, as `objective::look_offset`
    /// gives a built-in look's (temperature as a warm shift in mireds).
    fn offset(self) -> [f32; Params::COUNT] {
        match self {
            Style::Vivid => look_offset(&builtin("vivid")),
            Style::Moody => look_offset(&builtin("moody")),
            Style::Bright => look_offset(&builtin("bright")),
            Style::HighContrast => deltas(HIGH_CONTRAST),
            Style::Skin => deltas(SKIN),
            Style::BlackWhite | Style::Natural => [0.0; Params::COUNT],
        }
    }
}

/// High contrast: a steeper midtone and S-curve, deeper blacks and brighter
/// whites with the highlights held, a little more local contrast.
const HIGH_CONTRAST: &[(&str, f32)] = &[
    ("contrast", 30.0),
    ("highlights", -10.0),
    ("whites", 12.0),
    ("blacks", -15.0),
    ("curve_lights", 8.0),
    ("curve_darks", -8.0),
    ("clarity", 8.0),
];

/// Protect and warm skin: slightly warmer light, brighter and calmer
/// orange (skin) tones, smoother texture, vibrance rather than saturation.
const SKIN: &[(&str, f32)] = &[
    ("temperature", 6.0),
    ("tint", 2.0),
    ("vibrance", 6.0),
    ("texture", -6.0),
    ("clarity", -4.0),
    ("saturation_orange", -4.0),
    ("luminance_orange", 8.0),
    ("saturation_red", -3.0),
];

/// An offset array from (field, amount) pairs; temperature is a warm shift
/// in mireds. Fields this build does not have are skipped.
fn deltas(list: &[(&str, f32)]) -> [f32; Params::COUNT] {
    let mut out = [0.0; Params::COUNT];
    for &(key, v) in list {
        if let Some(i) = Params::NAMES.iter().position(|n| *n == key) {
            out[i] = v;
        }
    }
    out
}

fn moved(auto: &Params, offset: &[f32; Params::COUNT]) -> Params {
    offset_params(auto, offset, 1.0).quantized()
}

fn builtin(id: &str) -> LookSpec {
    LookSpec::builtin()
        .into_iter()
        .find(|l| l.id == id)
        .unwrap_or_else(LookSpec::natural)
}

/// Id of the look that mixes every style a request names.
const MIX_ID: &str = "request";

/// What keywords make of a request.
#[derive(Debug, Clone, Default)]
pub(super) struct Routed {
    /// The styles named, in [`Style::ALL`] order, each with the first
    /// keyword that named it.
    pub(super) found: Vec<(Style, &'static str)>,
}

/// The styles `request` names, by keyword.
pub(super) fn route(request: &str) -> Routed {
    let words = words(request);
    let found = Style::ALL
        .into_iter()
        .filter_map(|style| {
            let keyword = style
                .keywords()
                .iter()
                .find(|phrase| matches(&words, phrase))?;
            Some((style, *keyword))
        })
        .collect();
    Routed { found }
}

impl Routed {
    pub(super) fn has(&self, style: Style) -> bool {
        self.found.iter().any(|(s, _)| *s == style)
    }

    /// The styles beyond natural that the request names.
    fn styled(&self) -> Vec<Style> {
        self.found
            .iter()
            .map(|(s, _)| *s)
            .filter(|s| !matches!(s, Style::Natural | Style::BlackWhite))
            .collect()
    }

    /// The looks to search: natural (the baseline), each named style's
    /// look, and, when several styles beside black and white are named, one
    /// look that mixes them all.
    pub(super) fn looks(&self, auto: &Params) -> Vec<LookSpec> {
        let mut looks = vec![LookSpec::natural()];
        for (style, _) in &self.found {
            let look = style.look(auto);
            if !looks.iter().any(|l| l.id == look.id) {
                looks.push(look);
            }
        }
        let styled = self.styled();
        if styled.len() > 1 {
            let mut offset = [0.0; Params::COUNT];
            for style in &styled {
                for (o, d) in offset.iter_mut().zip(style.offset()) {
                    *o += d;
                }
            }
            let label = styled
                .iter()
                .map(|s| s.label())
                .collect::<Vec<_>>()
                .join(" + ");
            looks.push(LookSpec::proposal(MIX_ID, &label, moved(auto, &offset)));
        }
        looks
    }

    /// Id of the look that fulfils the request best by keywords: black and
    /// white when named, else the mix of several styles, the one style
    /// named, or natural; None when no keyword matched.
    pub(super) fn target(&self) -> Option<String> {
        if self.has(Style::BlackWhite) {
            return Some(LookSpec::black_and_white().id);
        }
        let styled = self.styled();
        match styled.as_slice() {
            [] if self.has(Style::Natural) => Some(LookSpec::natural().id),
            [] => None,
            [one] => Some(one.look(&Params::neutral(5500.0, 0.0)).id),
            _ => Some(MIX_ID.to_owned()),
        }
    }

    /// Whether `id` is one of the looks [`Routed::looks`] builds from auto
    /// (not a built-in look), by its candidate id ("look:contrast").
    pub(super) fn built(candidate_id: &str) -> bool {
        let id = candidate_id.strip_prefix("look:").unwrap_or(candidate_id);
        ["contrast", "skin", MIX_ID].contains(&id)
    }

    /// The honest account of a keyword-only run: `why` Claude did not
    /// follow the request, which words were understood, and what that
    /// chose.
    pub(super) fn note(&self, why: &str, chosen: Option<&str>) -> String {
        if self.found.is_empty() {
            return format!(
                "{why}, so only keywords in the request could be used, and none matched a known \
                 style (black and white, high contrast, vivid or saturated, natural, cinematic or \
                 moody, bright, skin). The usual edit was chosen."
            );
        }
        let understood = self.understood();
        match chosen {
            Some(label) => format!(
                "{why}, so only keywords in the request were used: {understood}. Chosen: {label}."
            ),
            None => format!("{why}, so only keywords in the request were used: {understood}."),
        }
    }

    /// The keywords read and the styles they named: "\"siyah beyaz\" as
    /// black and white, \"canli\" as vivid".
    pub(super) fn understood(&self) -> String {
        self.found
            .iter()
            .map(|(style, keyword)| {
                format!(
                    "\"{}\" as {}",
                    keyword.trim_end_matches('*'),
                    style.label().to_lowercase()
                )
            })
            .collect::<Vec<_>>()
            .join(", ")
    }
}

/// Lower-case words of `text` with Turkish letters folded to ASCII ("Yüksek
/// kontrastlı" is "yuksek kontrastli") and "&" read as "and" ("b&w").
fn words(text: &str) -> Vec<String> {
    let mut folded = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            'ç' | 'Ç' => folded.push('c'),
            'ğ' | 'Ğ' => folded.push('g'),
            'ı' | 'I' | 'İ' => folded.push('i'),
            'ö' | 'Ö' => folded.push('o'),
            'ş' | 'Ş' => folded.push('s'),
            'ü' | 'Ü' => folded.push('u'),
            'â' | 'Â' => folded.push('a'),
            'î' | 'Î' => folded.push('i'),
            'û' | 'Û' => folded.push('u'),
            '&' => folded.push_str(" and "),
            // The dot a lower-cased "İ" may carry.
            '\u{307}' => {}
            c => folded.extend(c.to_lowercase()),
        }
    }
    folded
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_owned)
        .collect()
}

/// Whether `phrase` (space-separated words, the last may end in `*`)
/// occurs as consecutive `words`.
fn matches(words: &[String], phrase: &str) -> bool {
    let parts: Vec<&str> = phrase.split(' ').collect();
    words.windows(parts.len()).any(|window| {
        window
            .iter()
            .zip(&parts)
            .all(|(word, part)| match part.strip_suffix('*') {
                Some(stem) => word.starts_with(stem),
                None => word == part,
            })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn styles(request: &str) -> Vec<Style> {
        route(request).found.into_iter().map(|(s, _)| s).collect()
    }

    #[test]
    fn keywords_route_turkish_and_english() {
        use Style::*;
        for (request, want) in [
            (
                "siyah beyaz, içeriğe göre en iyi siyah beyaz",
                &[BlackWhite][..],
            ),
            ("SİYAH BEYAZ", &[BlackWhite]),
            ("Black & White please", &[BlackWhite]),
            ("b&w", &[BlackWhite]),
            ("black-and-white, moody", &[BlackWhite, Moody]),
            ("monochrome", &[BlackWhite]),
            ("high contrast, saturated", &[HighContrast, Vivid]),
            ("yüksek kontrastlı ve doygun", &[HighContrast, Vivid]),
            ("canlı renkler", &[Vivid]),
            ("optimum colour distribution", &[Natural]),
            ("doğal", &[Natural]),
            ("sinematik", &[Moody]),
            ("cinematic / motion-like", &[Moody]),
            ("maximize skin colour", &[Skin]),
            ("ten rengi güzel olsun", &[Skin]),
            ("light and airy", &[Bright]),
            ("make it pop somehow", &[]),
            // Whole words only: "ten" is not in "içten", "mono" not in "monotone".
            ("içten bir his, monotone değil", &[]),
        ] {
            assert_eq!(styles(request), want, "{request:?}");
        }
    }

    #[test]
    fn black_and_white_requests_target_the_black_and_white_look() {
        let auto = Params::neutral(5500.0, 0.0);
        let routed = route("siyah beyaz, içeriğe göre en iyi siyah beyaz");
        let bw = LookSpec::black_and_white();
        assert_eq!(routed.target(), Some(bw.id.clone()));
        let looks = routed.looks(&auto);
        let ids: Vec<&str> = looks.iter().map(|l| l.id.as_str()).collect();
        assert_eq!(ids, ["natural", bw.id.as_str()]);
        // Black and white wins over any other style it is named with.
        let routed = route("high contrast black and white");
        assert_eq!(routed.target(), Some(bw.id));
    }

    #[test]
    fn several_styles_get_one_look_that_mixes_them() {
        let auto = Params {
            contrast: 10.0,
            ..Params::neutral(5500.0, 0.0)
        };
        let routed = route("high contrast, saturated");
        assert_eq!(routed.target().as_deref(), Some(MIX_ID));
        let looks = routed.looks(&auto);
        let ids: Vec<&str> = looks.iter().map(|l| l.id.as_str()).collect();
        assert_eq!(ids, ["natural", "contrast", "vivid", MIX_ID]);
        let mix = looks.last().unwrap();
        assert_eq!(mix.label, "High contrast + Vivid");
        let anchor = mix.anchor.unwrap();
        // High contrast's +30 and vivid's +18 on auto's +10.
        assert_eq!(anchor.contrast, 58.0);
        assert!(anchor.vibrance > 0.0 && anchor.clarity > 0.0);
        let contrast = looks[1].anchor.unwrap();
        assert_eq!((contrast.contrast, contrast.blacks), (40.0, -15.0));
        assert!(Routed::built("look:contrast") && Routed::built("look:request"));
        assert!(!Routed::built("look:vivid"));
        // One style: its own look; natural alone: natural; nothing: none.
        assert_eq!(route("vivid").target().as_deref(), Some("vivid"));
        assert_eq!(route("skin").target().as_deref(), Some("skin"));
        assert_eq!(route("doğal olsun").target().as_deref(), Some("natural"));
        assert_eq!(route("hmm").target(), None);
        assert_eq!(route("hmm").looks(&auto).len(), 1);
    }

    #[test]
    fn notes_say_only_keywords_were_used() {
        let note =
            route("yüksek kontrast, canlı").note("Claude was off", Some("High contrast + Vivid"));
        assert_eq!(
            note,
            "Claude was off, so only keywords in the request were used: \"yuksek kontrast\" as \
             high contrast, \"canli\" as vivid. Chosen: High contrast + Vivid."
        );
        let none = route("something lovely").note("Claude was off", None);
        assert!(none.contains("none matched"), "{none}");
    }
}
