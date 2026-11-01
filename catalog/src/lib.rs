//! The one list of apps a host links in. Adding an app = one feature in
//! Cargo.toml plus one line here; nothing else in the platform changes.

use spl_core::Registry;

pub fn registry() -> Registry {
    #[allow(unused_mut)]
    let mut registry = Registry::new();
    #[cfg(feature = "device-info")]
    spl_app_device_info::register(&mut registry);
    registry
}
