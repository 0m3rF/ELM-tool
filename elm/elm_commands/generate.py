import click
import pandas as pd
from elm.core import generation as core_gen

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
    # Parse columns if provided
    column_list = None
    if columns:
        column_list = [col.strip() for col in columns.split(',')]

    # Prepare pattern dictionary if pattern is provided
    pattern_dict = None
    if pattern and column_list:
        pattern_dict = {col: pattern for col in column_list}

    # Use core module to generate and save data
    result = core_gen.generate_and_save(
        num_records=num_records,
        columns=column_list,
        environment=environment,
        table=table,
        output_file=output,
        file_format=format.lower(),
        write_to_db=write_to_db,
        mode=mode,
        string_length=string_length,
        pattern=pattern_dict,
        min_number=min_number,
        max_number=max_number,
        decimal_places=decimal_places,
        start_date=start_date,
        end_date=end_date,
        date_format=date_format
    )

    if result.success:
        click.echo(result.message)
        if not output and not write_to_db and result.data:
            # Print to console if no output specified
            df = pd.DataFrame(result.data)
            if format.upper() == 'JSON':
                click.echo(df.to_json(orient='records', indent=2))
            else:
                click.echo(df.to_string(index=False))
    else:
        raise click.UsageError(result.message)

# Define command aliases
ALIASES = {
    "d": data,
    "random": data,
    "rand": data
}