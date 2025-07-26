import click
import pandas as pd
from elm.elm_utils.random_data import generate_random_data
from elm.elm_utils.db_utils import get_connection_url, check_table_exists, get_table_columns, write_to_db, write_to_file

class AliasedGroup(click.Group):
    def get_command(self, ctx, cmd_name):
        try:
            cmd_name = ALIASES[cmd_name].name
        except KeyError:
            pass
        return super().get_command(ctx, cmd_name)

@click.group(cls=AliasedGroup)
def generate():
    """Data generation commands for testing"""
    pass

@generate.command()
@click.option("-n", "--num-records", type=int, default=10, help="Number of records to generate")
@click.option("-c", "--columns", help="Comma-separated list of column names (if not specified, uses all columns from table)")
@click.option("-e", "--environment", help="Environment name to get table schema from")
@click.option("-t", "--table", help="Table name to get schema from")
@click.option("-o", "--output", help="Output file path (if not specified, prints to console)")
@click.option("-f", "--format", type=click.Choice(['CSV', 'JSON'], case_sensitive=False), default='CSV', help="Output file format")
@click.option("--string-length", type=int, default=10, help="Default length for string values")
@click.option("--pattern", help="Pattern for string generation (email, name, address, phone, ssn, username, url, ipv4, ipv6, uuid)")
@click.option("--min-number", type=int, default=0, help="Minimum value for number generation")
@click.option("--max-number", type=int, default=1000, help="Maximum value for number generation")
@click.option("--decimal-places", type=int, default=0, help="Decimal places for number generation")
@click.option("--start-date", help="Start date for date generation (YYYY-MM-DD)")
@click.option("--end-date", help="End date for date generation (YYYY-MM-DD)")
@click.option("--date-format", default="%Y-%m-%d", help="Date format for date generation")
@click.option("--write-to-db", is_flag=True, help="Write generated data to database table")
@click.option("--mode", type=click.Choice(['APPEND', 'REPLACE', 'FAIL'], case_sensitive=False), default='APPEND', help="Table write mode when writing to database")
def data(num_records, columns, environment, table, output, format, string_length, pattern,
        min_number, max_number, decimal_places, start_date, end_date, date_format, write_to_db, mode):
    """Generate random data for testing

    Examples:

        Generate 10 random records for specified columns:
          elm-tool generate data --columns "id,name,email,created_at" --num-records 10

        Generate data based on table schema:
          elm-tool generate data --environment dev --table users --num-records 100

        Generate data with specific patterns:
          elm-tool generate data --columns "id,name,email" --pattern "email" --num-records 5

        Generate data and save to file:
          elm-tool generate data --columns "id,name,email" --output "test_data.csv" --num-records 20

        Generate data with specific ranges:
          elm-tool generate data --columns "id,price,created_at" --min-number 100 --max-number 999 --start-date "2023-01-01" --end-date "2023-12-31"

        Generate data and write to database:
          elm-tool generate data --environment dev --table users --num-records 50 --write-to-db
    """
    try:
        # Parse columns if provided
        column_list = []
        if columns:
            column_list = [col.strip() for col in columns.split(',')]

        # Get schema from database if environment and table are provided
        if environment and table:
            # Get connection URL
            connection_url = get_connection_url(environment)

            # Check if table exists
            if not check_table_exists(connection_url, table):
                click.echo(f"Table '{table}' does not exist in environment '{environment}'")
                return

            # Get table columns
            db_columns = get_table_columns(connection_url, table)
            if not db_columns:
                click.echo(f"Could not retrieve columns for table '{table}'")
                return

            # Use provided columns or all columns from table
            if not column_list:
                column_list = db_columns
            else:
                # Validate that all provided columns exist in the table
                missing_columns = set(column_list) - set(db_columns)
                if missing_columns:
                    click.echo(f"The following columns do not exist in table '{table}': {', '.join(missing_columns)}")
                    return

        # Ensure we have columns to generate data for
        if not column_list:
            click.echo("No columns specified. Please provide columns or a table schema.")
            return

        # Prepare column parameters
        column_params = {}
        for column in column_list:
            column_params[column] = {
                'type': None,  # Will be inferred
                'length': string_length,
                'pattern': pattern,
                'min_val': min_number,
                'max_val': max_number,
                'decimal_places': decimal_places,
                'start_date': start_date,
                'end_date': end_date,
                'date_format': date_format
            }

        # Generate random data
        click.echo(f"Generating {num_records} random records for columns: {', '.join(column_list)}")
        data = generate_random_data(column_list, num_records, **column_params)

        # Write to database if requested
        if write_to_db and environment and table:
            # Map mode to SQLAlchemy if_exists parameter
            if_exists_map = {
                'APPEND': 'append',
                'REPLACE': 'replace',
                'FAIL': 'fail'
            }
            if_exists = if_exists_map[mode.upper()]

            # Write to database
            write_to_db(data, connection_url, table, if_exists)
            click.echo(f"Successfully wrote {num_records} records to table '{table}' in environment '{environment}'")

        # Write to file if output is provided
        elif output:
            # Write to file
            write_to_file(data, output, format.lower())
            click.echo(f"Successfully wrote {num_records} records to file '{output}'")

        # Otherwise, print to console
        else:
            # Print to console (limit to 10 records for readability)
            display_data = data.head(10) if len(data) > 10 else data
            if format.upper() == 'JSON':
                click.echo(display_data.to_json(orient='records', indent=2))
            else:
                click.echo(display_data.to_string(index=False))

            if len(data) > 10:
                click.echo(f"\n... and {len(data) - 10} more records (showing first 10 only)")

    except Exception as e:
        click.echo(f"Error generating random data: {str(e)}")

# Define command aliases
ALIASES = {
    "d": data,
    "random": data,
    "rand": data
}