/// One schema step. Every backend lists the same versions and names.
#[derive(Debug, Clone, Copy)]
pub struct Migration {
    pub version: i64,
    pub name: &'static str,
    pub sql: &'static str,
}

macro_rules! migrations {
    ($dialect:literal: $($version:literal $name:literal),+ $(,)?) => {
        &[$(Migration {
            version: $version,
            name: $name,
            sql: include_str!(concat!("../migrations/", $dialect, "/", $name, ".sql")),
        }),+]
    };
}

// `<version> "<file stem>"`; stems are zero-padded so the files sort.
pub const SQLITE: &[Migration] = migrations!("sqlite": 1 "0001_init", 2 "0002_announcements");
pub const POSTGRES: &[Migration] = migrations!("postgres": 1 "0001_init", 2 "0002_announcements");
pub const ORACLE: &[Migration] = migrations!("oracle": 1 "0001_init", 2 "0002_announcements");

#[cfg(test)]
mod tests {
    use super::*;

    fn files(dialect: &str) -> Vec<String> {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/migrations/");
        let mut names: Vec<String> = std::fs::read_dir(format!("{dir}{dialect}"))
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        names
    }

    #[test]
    fn every_backend_has_the_same_migrations() {
        let sqlite = files("sqlite");
        assert_eq!(sqlite, files("postgres"), "postgres migrations out of step");
        assert_eq!(sqlite, files("oracle"), "oracle migrations out of step");
        assert_eq!(
            sqlite.len(),
            SQLITE.len(),
            "a migration file is not registered"
        );
        for set in [POSTGRES, ORACLE] {
            let a: Vec<_> = SQLITE.iter().map(|m| (m.version, m.name)).collect();
            let b: Vec<_> = set.iter().map(|m| (m.version, m.name)).collect();
            assert_eq!(a, b);
        }
    }
}
