# External Integrations

## Database Systems
The ELM tool integrates with various internal relational database management systems using `sqlalchemy` and specific drivers:
- **PostgreSQL** (`psycopg2-binary` integration)
- **Oracle** (`oracledb` integration)
- **MySQL / MariaDB** (`pymysql` integration)
- **Microsoft SQL Server** (`pyodbc` integration)

## External APIs
Currently, the ELM tool focuses primarily on internal data extraction, loading, and masking across relational databases. There are no explicit third-party REST/GraphQL APIs (like Stripe or social logins) configured out of the box. 

## Operating System Context
- Integrates with the host OS filesystem via `platformdirs` to store configurations and environments.
- Relies heavily on command-line context (`click`) and standard inputs/outputs.
