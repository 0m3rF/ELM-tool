//! The masking rule CRUD surface (`mask.add`/`mask.edit`/`mask.list`/`mask.remove`/`mask.test`)
//! backs the desktop "Masking rules" screen and `elm mask ...`, but had no test coverage of any
//! kind before this file: every existing masking test exercised `apply_masks`/`mask_text`
//! directly, never the daemon operations a real client actually calls.

use elm_core::{
    Identifier, MaskAlgorithm, MaskRule, MaskRuleId,
    protocol::{Operation, Response},
};
use elm_daemon::{DaemonClient, DaemonRuntime, RuntimePaths};
use tempfile::tempdir;

#[tokio::test]
async fn mask_rules_round_trip_through_add_edit_test_and_remove() {
    let directory = tempdir().unwrap_or_else(|error| panic!("{error}"));
    let paths = RuntimePaths::for_data_directory(directory.path().join("state"))
        .unwrap_or_else(|error| panic!("{error}"));
    let runtime = DaemonRuntime::open(paths.clone()).unwrap_or_else(|error| panic!("{error}"));
    let daemon = tokio::spawn(runtime.serve());
    let client = DaemonClient::connect(paths).unwrap_or_else(|error| panic!("{error}"));

    let rule = MaskRule {
        id: MaskRuleId::new(),
        name: "ssn".into(),
        column: Identifier::new("ssn").unwrap_or_else(|error| panic!("{error}")),
        algorithm: MaskAlgorithm::Star,
        environment_id: None,
    };

    match client
        .request(Operation::MaskAdd(rule.clone()))
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::Masks(saved) => assert_eq!(saved, vec![rule.clone()]),
        other => panic!("expected a masks response, got {other:?}"),
    }

    match client
        .request(Operation::MaskList)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::Masks(rules) => assert_eq!(rules, vec![rule.clone()]),
        other => panic!("expected a masks response, got {other:?}"),
    }

    match client
        .request(Operation::MaskTest {
            rule: rule.clone(),
            value: "123-45-6789".into(),
        })
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::MaskedValue(value) => assert_eq!(value, "*****"),
        other => panic!("expected a masked value response, got {other:?}"),
    }

    // Editing keeps the same id (the desktop "Edit masking rule" flow round-trips the id from
    // the list) but changes the algorithm; `mask.list` and a fresh `mask.test` must reflect it.
    let mut edited = rule.clone();
    edited.name = "ssn (last 4 visible)".into();
    edited.algorithm = MaskAlgorithm::StarLength { unmasked_prefix: 0 };
    match client
        .request(Operation::MaskEdit(edited.clone()))
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::Masks(saved) => assert_eq!(saved, vec![edited.clone()]),
        other => panic!("expected a masks response, got {other:?}"),
    }
    match client
        .request(Operation::MaskList)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::Masks(rules) => assert_eq!(rules, vec![edited.clone()]),
        other => panic!("expected a masks response, got {other:?}"),
    }
    match client
        .request(Operation::MaskTest {
            rule: edited.clone(),
            value: "123-45-6789".into(),
        })
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::MaskedValue(value) => assert_eq!(value, "***********"),
        other => panic!("expected a masked value response, got {other:?}"),
    }

    client
        .request(Operation::MaskRemove(edited.id))
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    match client
        .request(Operation::MaskList)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
    {
        Response::Masks(rules) => assert!(rules.is_empty()),
        other => panic!("expected a masks response, got {other:?}"),
    }
    let remove_again = client.request(Operation::MaskRemove(edited.id)).await;
    assert!(
        matches!(remove_again, Err(elm_core::ElmError::NotFound(_))),
        "removing an already-removed rule should fail closed, got {remove_again:?}"
    );

    client
        .request(Operation::DaemonStop)
        .await
        .unwrap_or_else(|error| panic!("{error}"));
    tokio::time::timeout(std::time::Duration::from_secs(5), daemon)
        .await
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|error| panic!("{error}"))
        .unwrap_or_else(|error| panic!("{error}"));
}
