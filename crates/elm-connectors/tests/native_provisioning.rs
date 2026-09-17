#![cfg(all(
    target_os = "windows",
    feature = "native-client-provisioning",
    feature = "oracle"
))]

use elm_connectors::native_provisioning::{
    oracle_instant_client_present, provision_oracle_instant_client,
};

/// Live-tested: downloads the real, pinned Oracle Instant Client Basic ZIP, verifies its
/// SHA-256, extracts it next to this test binary, and confirms `oracle::Version::client()` -- the
/// same probe every real connector call makes -- can load it afterward without any environment
/// variable or elevation. Cleans up the extracted files it can; `oci.dll` and its own dependency
/// `oraociei.dll` stay loaded for this process's lifetime once `oracle::Version::client()`
/// succeeds, so Windows refuses to delete them until the test process exits -- a real "cargo
/// test" run leaves those two files in `target/debug/deps/` for the next run to overwrite,
/// which is harmless.
#[tokio::test]
#[ignore = "downloads a real ~90 MB archive from Oracle"]
async fn provisioning_makes_the_oracle_client_library_loadable() {
    assert!(
        !oracle_instant_client_present(),
        "this test assumes no Oracle client is already on this host/next to the test binary; \
         found one already present, so the before/after comparison below would be meaningless"
    );

    let destination = provision_oracle_instant_client()
        .await
        .unwrap_or_else(|error| panic!("provisioning failed: {error}"));

    assert!(
        oracle_instant_client_present(),
        "oracle::Version::client() should succeed once Instant Client is next to the exe"
    );

    for entry in std::fs::read_dir(&destination).unwrap_or_else(|error| panic!("{error}")) {
        let entry = entry.unwrap_or_else(|error| panic!("{error}"));
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if name.ends_with(".dll")
            || name.ends_with(".sym")
            || name.ends_with(".exe")
            || name.ends_with(".jar")
            || name == "BASIC_LICENSE"
            || name == "BASIC_README"
        {
            let _ = std::fs::remove_file(entry.path());
        }
    }
}
