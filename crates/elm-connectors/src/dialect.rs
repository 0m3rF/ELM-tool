use elm_core::{DatabaseKind, Identifier, Relation, WriteMode};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicationPlan {
    pub create_staging: String,
    pub publish: Vec<String>,
    pub cleanup: String,
}

pub trait Dialect: Send + Sync {
    fn kind(&self) -> DatabaseKind;
    fn quote_identifier(&self, identifier: &Identifier) -> String;
    fn parameter(&self, index: usize) -> String;

    fn quote_relation(&self, relation: &Relation) -> String {
        [
            relation.catalog.as_ref(),
            relation.schema.as_ref(),
            Some(&relation.name),
        ]
        .into_iter()
        .flatten()
        .map(|identifier| self.quote_identifier(identifier))
        .collect::<Vec<_>>()
        .join(".")
    }

    fn publication_plan(
        &self,
        target: &Relation,
        staging: &Relation,
        backup: &Relation,
        mode: WriteMode,
    ) -> PublicationPlan;
}

#[derive(Debug)]
struct PostgreSqlDialect;
#[derive(Debug)]
struct MySqlDialect;
#[derive(Debug)]
struct SqlServerDialect;
#[derive(Debug)]
struct OracleDialect;

pub fn dialect_for(kind: DatabaseKind) -> &'static dyn Dialect {
    static POSTGRESQL: PostgreSqlDialect = PostgreSqlDialect;
    static MYSQL: MySqlDialect = MySqlDialect;
    static SQL_SERVER: SqlServerDialect = SqlServerDialect;
    static ORACLE: OracleDialect = OracleDialect;
    match kind {
        DatabaseKind::PostgreSql => &POSTGRESQL,
        DatabaseKind::MySql => &MYSQL,
        DatabaseKind::SqlServer => &SQL_SERVER,
        DatabaseKind::Oracle => &ORACLE,
    }
}

fn double_quote(identifier: &Identifier) -> String {
    format!("\"{}\"", identifier.as_str().replace('"', "\"\""))
}

fn append_plan(dialect: &dyn Dialect, target: &Relation, staging: &Relation) -> Vec<String> {
    vec![format!(
        "INSERT INTO {} SELECT * FROM {}",
        dialect.quote_relation(target),
        dialect.quote_relation(staging)
    )]
}

impl Dialect for PostgreSqlDialect {
    fn kind(&self) -> DatabaseKind {
        DatabaseKind::PostgreSql
    }

    fn quote_identifier(&self, identifier: &Identifier) -> String {
        double_quote(identifier)
    }

    fn parameter(&self, index: usize) -> String {
        format!("${index}")
    }

    fn publication_plan(
        &self,
        target: &Relation,
        staging: &Relation,
        backup: &Relation,
        mode: WriteMode,
    ) -> PublicationPlan {
        let target_name = self.quote_identifier(&target.name);
        let backup_name = self.quote_identifier(&backup.name);
        let publish = match mode {
            WriteMode::Append => append_plan(self, target, staging),
            WriteMode::Replace => vec![
                format!(
                    "ALTER TABLE {} RENAME TO {backup_name}",
                    self.quote_relation(target)
                ),
                format!(
                    "ALTER TABLE {} RENAME TO {target_name}",
                    self.quote_relation(staging)
                ),
                format!(
                    "DROP TABLE {}",
                    relation_with_name(self, backup, &backup.name)
                ),
            ],
            WriteMode::Fail => vec![format!(
                "ALTER TABLE {} RENAME TO {target_name}",
                self.quote_relation(staging)
            )],
        };
        PublicationPlan {
            create_staging: format!(
                "CREATE TABLE {} (LIKE {} INCLUDING ALL)",
                self.quote_relation(staging),
                self.quote_relation(target)
            ),
            publish,
            cleanup: format!(
                "DROP TABLE IF EXISTS {}",
                relation_with_name(self, staging, &staging.name)
            ),
        }
    }
}

impl Dialect for MySqlDialect {
    fn kind(&self) -> DatabaseKind {
        DatabaseKind::MySql
    }

    fn quote_identifier(&self, identifier: &Identifier) -> String {
        format!("`{}`", identifier.as_str().replace('`', "``"))
    }

    fn parameter(&self, _index: usize) -> String {
        "?".into()
    }

    fn publication_plan(
        &self,
        target: &Relation,
        staging: &Relation,
        backup: &Relation,
        mode: WriteMode,
    ) -> PublicationPlan {
        let publish = match mode {
            WriteMode::Append => append_plan(self, target, staging),
            WriteMode::Replace => vec![format!(
                "RENAME TABLE {} TO {}, {} TO {}",
                self.quote_relation(target),
                self.quote_relation(backup),
                self.quote_relation(staging),
                self.quote_relation(target)
            )],
            WriteMode::Fail => vec![format!(
                "RENAME TABLE {} TO {}",
                self.quote_relation(staging),
                self.quote_relation(target)
            )],
        };
        PublicationPlan {
            create_staging: format!(
                "CREATE TABLE {} LIKE {}",
                self.quote_relation(staging),
                self.quote_relation(target)
            ),
            publish,
            cleanup: format!("DROP TABLE IF EXISTS {}", self.quote_relation(staging)),
        }
    }
}

impl Dialect for SqlServerDialect {
    fn kind(&self) -> DatabaseKind {
        DatabaseKind::SqlServer
    }

    fn quote_identifier(&self, identifier: &Identifier) -> String {
        format!("[{}]", identifier.as_str().replace(']', "]]"))
    }

    fn parameter(&self, _index: usize) -> String {
        "?".into()
    }

    fn publication_plan(
        &self,
        target: &Relation,
        staging: &Relation,
        backup: &Relation,
        mode: WriteMode,
    ) -> PublicationPlan {
        let publish = match mode {
            WriteMode::Append => append_plan(self, target, staging),
            WriteMode::Replace => vec![
                format!(
                    "EXEC sp_rename N'{}', N'{}'",
                    self.quote_relation(target),
                    backup.name.as_str().replace("'", "''")
                ),
                format!(
                    "EXEC sp_rename N'{}', N'{}'",
                    self.quote_relation(staging),
                    target.name.as_str().replace("'", "''")
                ),
            ],
            WriteMode::Fail => vec![format!(
                "EXEC sp_rename N'{}', N'{}'",
                self.quote_relation(staging),
                target.name.as_str().replace("'", "''")
            )],
        };
        PublicationPlan {
            create_staging: format!(
                "SELECT TOP (0) * INTO {} FROM {}",
                self.quote_relation(staging),
                self.quote_relation(target)
            ),
            publish,
            cleanup: format!(
                "IF OBJECT_ID(N'{}') IS NOT NULL DROP TABLE {}",
                self.quote_relation(staging).replace("'", "''"),
                self.quote_relation(staging)
            ),
        }
    }
}

impl Dialect for OracleDialect {
    fn kind(&self) -> DatabaseKind {
        DatabaseKind::Oracle
    }

    fn quote_identifier(&self, identifier: &Identifier) -> String {
        double_quote(identifier)
    }

    fn parameter(&self, index: usize) -> String {
        format!(":{index}")
    }

    fn publication_plan(
        &self,
        target: &Relation,
        staging: &Relation,
        _backup: &Relation,
        mode: WriteMode,
    ) -> PublicationPlan {
        // This generic plan cannot express a journaled non-atomic swap.
        // OracleSink owns that explicit mode; never emit an incomplete online
        // redefinition call or an unjournaled two-rename fallback here.
        let publish = match mode {
            WriteMode::Append => append_plan(self, target, staging),
            WriteMode::Replace => vec![
                "BEGIN RAISE_APPLICATION_ERROR(-20002, 'Oracle REPLACE requires the journaled table-swap connector'); END;".into(),
            ],
            WriteMode::Fail => vec![format!(
                "RENAME {} TO {}",
                self.quote_relation(staging),
                self.quote_identifier(&target.name)
            )],
        };
        PublicationPlan {
            create_staging: format!(
                "CREATE TABLE {} AS SELECT * FROM {} WHERE 1 = 0",
                self.quote_relation(staging),
                self.quote_relation(target)
            ),
            publish,
            cleanup: format!("DROP TABLE {} PURGE", self.quote_relation(staging)),
        }
    }
}

fn relation_with_name(dialect: &dyn Dialect, relation: &Relation, name: &Identifier) -> String {
    let relation = Relation {
        catalog: relation.catalog.clone(),
        schema: relation.schema.clone(),
        name: name.clone(),
    };
    dialect.quote_relation(&relation)
}

#[cfg(test)]
mod tests {
    use elm_core::{DatabaseKind, Identifier, Relation};

    use super::dialect_for;

    #[test]
    fn quotes_reserved_and_mixed_case_identifiers() {
        let relation = Relation {
            catalog: None,
            schema: Some(Identifier::new("Mixed Schema").unwrap_or_else(|error| panic!("{error}"))),
            name: Identifier::new("select\"ions").unwrap_or_else(|error| panic!("{error}")),
        };
        assert_eq!(
            dialect_for(DatabaseKind::PostgreSql).quote_relation(&relation),
            "\"Mixed Schema\".\"select\"\"ions\""
        );
        assert_eq!(
            dialect_for(DatabaseKind::SqlServer).quote_identifier(
                &Identifier::new("a]b").unwrap_or_else(|error| panic!("{error}"))
            ),
            "[a]]b]"
        );
    }

    #[test]
    fn checkpoint_parameters_are_never_literals() {
        assert_eq!(dialect_for(DatabaseKind::PostgreSql).parameter(2), "$2");
        assert_eq!(dialect_for(DatabaseKind::Oracle).parameter(1), ":1");
    }
}
