//! Best-effort physical-identity discovery for database source/sink relations.
//!
//! This never gates, blocks, or changes a transfer: staged publication is already
//! safe whether or not the source and destination are the same physical table, because
//! the source is always fully staged before the destination is touched (see the
//! `same_table` and `interrupted_same_table` cross-database fixtures). This module exists
//! only so a job can report an accurate warning when a source and destination reached
//! through different names or different configured hosts/ports — a view, a synonym, or a
//! second environment record for the same physical server — turn out to be the same
//! underlying table.
//!
//! Resolution unwinds at most one level of view or synonym indirection and never guesses:
//! an ambiguous view (more than one underlying table), a synonym across a database link,
//! or a server fingerprint the connected role cannot read all resolve to `Ok(None)`. Callers
//! must treat `None` as "unknown", never as "confirmed different".

#[cfg(any(
    feature = "postgresql",
    feature = "mysql",
    feature = "sql-server",
    feature = "oracle"
))]
use elm_core::DatabaseKind;
use elm_core::{Environment, Relation, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhysicalIdentity {
    /// Identifies the physical server/instance independent of the hostname or port used to
    /// reach it. Absent (rather than guessed) whenever the connected role cannot read a
    /// sufficiently unique server property.
    pub server_fingerprint: String,
    /// The database/catalog name as the connected session sees it.
    pub database: String,
    pub base_schema: Option<String>,
    pub base_relation: String,
}

/// Resolves the physical identity of `relation` on `environment`, unwinding a single level
/// of view or synonym indirection through catalog metadata (never by parsing SQL text).
///
/// Returns `Ok(None)` whenever identity cannot be established conservatively: the relation
/// does not exist, is a view/synonym with more than one distinct underlying table, is a
/// synonym across a database link, or the connected role cannot read the server fingerprint.
pub async fn resolve(
    environment: &Environment,
    password: &str,
    relation: &Relation,
) -> Result<Option<PhysicalIdentity>> {
    match environment.kind {
        #[cfg(feature = "postgresql")]
        DatabaseKind::PostgreSql => {
            crate::postgresql::resolve_physical_identity(environment, password, relation).await
        }
        #[cfg(feature = "mysql")]
        DatabaseKind::MySql => {
            crate::mysql::resolve_physical_identity(environment, password, relation).await
        }
        #[cfg(feature = "sql-server")]
        DatabaseKind::SqlServer => {
            crate::sql_server::resolve_physical_identity(environment, password, relation).await
        }
        #[cfg(feature = "oracle")]
        DatabaseKind::Oracle => {
            crate::oracle_source::resolve_physical_identity(environment, password, relation).await
        }
        #[allow(unreachable_patterns)]
        _ => {
            let _ = (password, relation);
            Ok(None)
        }
    }
}

/// A human-readable warning for a job whose source and destination resolved to the same
/// physical table, naming how each side reached it.
#[must_use]
pub fn same_table_warning(source_relation: &Relation, sink_relation: &Relation) -> String {
    format!(
        "source '{}' and destination '{}' resolve to the same physical table; staged \
         publication already isolates the copy from this table before the destination is \
         touched, but review whether this transfer was intended",
        display_relation(source_relation),
        display_relation(sink_relation)
    )
}

fn display_relation(relation: &Relation) -> String {
    [
        relation.catalog.as_ref().map(|value| value.as_str()),
        relation.schema.as_ref().map(|value| value.as_str()),
        Some(relation.name.as_str()),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>()
    .join(".")
}

impl PhysicalIdentity {
    #[must_use]
    pub fn same_physical_table(&self, other: &Self) -> bool {
        self == other
    }
}

#[cfg(test)]
mod tests {
    use elm_core::Identifier;

    use super::*;

    fn relation(schema: Option<&str>, name: &str) -> Relation {
        Relation {
            catalog: None,
            schema: schema
                .map(|value| Identifier::new(value).unwrap_or_else(|error| panic!("{error}"))),
            name: Identifier::new(name).unwrap_or_else(|error| panic!("{error}")),
        }
    }

    #[test]
    fn identical_identities_match_regardless_of_field_order() {
        let a = PhysicalIdentity {
            server_fingerprint: "fp".into(),
            database: "db".into(),
            base_schema: Some("public".into()),
            base_relation: "orders".into(),
        };
        let b = a.clone();
        assert!(a.same_physical_table(&b));
    }

    #[test]
    fn different_fingerprints_never_match() {
        let a = PhysicalIdentity {
            server_fingerprint: "fp-1".into(),
            database: "db".into(),
            base_schema: Some("public".into()),
            base_relation: "orders".into(),
        };
        let b = PhysicalIdentity {
            server_fingerprint: "fp-2".into(),
            ..a.clone()
        };
        assert!(!a.same_physical_table(&b));
    }

    #[test]
    fn warning_names_both_sides() {
        let warning = same_table_warning(
            &relation(Some("s"), "orders"),
            &relation(None, "orders_view"),
        );
        assert!(warning.contains("s.orders"));
        assert!(warning.contains("orders_view"));
    }
}
