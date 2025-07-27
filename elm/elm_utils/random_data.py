import random
import string
import re
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker

# Initialize Faker
fake = Faker()

def generate_random_string(length=10, pattern=None):
    """Generate a random string with the specified length or pattern"""
    if pattern:
        # Use regex pattern to generate string
        try:
            # Simple pattern handling for common cases
            if pattern == 'email':
                return fake.email()
            elif pattern == 'name':
                return fake.name()
            elif pattern == 'address':
                return fake.address().replace('\n', ', ')
            elif pattern == 'phone':
                return fake.phone_number()
            elif pattern == 'ssn':
                return fake.ssn()
            elif pattern == 'username':
                return fake.user_name()
            elif pattern == 'url':
                return fake.url()
            elif pattern == 'ipv4':
                return fake.ipv4()
            elif pattern == 'ipv6':
                return fake.ipv6()
            elif pattern == 'uuid':
                return str(fake.uuid4())
            elif pattern.startswith('regex:'):
                # Advanced regex pattern handling would go here
                # This is a simplified version
                regex_pattern = pattern[6:]  # Remove 'regex:' prefix
                if regex_pattern == r'\d{3}-\d{2}-\d{4}':  # SSN pattern
                    return f"{random.randint(100, 999)}-{random.randint(10, 99)}-{random.randint(1000, 9999)}"
                elif regex_pattern == r'\d{3}-\d{3}-\d{4}':  # Phone pattern
                    return f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
                else:
                    return fake.pystr(min_chars=length, max_chars=length)
            else:
                return fake.pystr(min_chars=length, max_chars=length)
        except Exception as e:
            print(f"Error generating string with pattern '{pattern}': {str(e)}")
            return generate_random_string(length)
    else:
        # Generate random string of specified length
        chars = string.ascii_letters + string.digits
        return ''.join(random.choice(chars) for _ in range(length))

def generate_random_number(min_val=0, max_val=1000, decimal_places=0):
    """Generate a random number within the specified range"""
    if decimal_places > 0:
        # Generate a random float
        value = random.uniform(min_val, max_val)
        # Round to specified decimal places
        return round(value, decimal_places)
    else:
        # Generate a random integer
        return random.randint(min_val, max_val)

def generate_random_date(start_date=None, end_date=None, date_format='%Y-%m-%d'):
    """Generate a random date within the specified range"""
    if not start_date:
        start_date = datetime(1970, 1, 1)
    elif isinstance(start_date, str):
        try:
            start_date = datetime.strptime(start_date, date_format)
        except ValueError:
            start_date = datetime(1970, 1, 1)

    if not end_date:
        end_date = datetime.now()
    elif isinstance(end_date, str):
        try:
            end_date = datetime.strptime(end_date, date_format)
        except ValueError:
            end_date = datetime.now()

    # Calculate the difference in days
    delta_days = (end_date - start_date).days
    if delta_days <= 0:
        delta_days = 365  # Default to 1 year if invalid range

    # Generate a random number of days to add
    random_days = random.randint(0, delta_days)

    # Add the random days to the start date
    random_date = start_date + timedelta(days=random_days)

    # Format the date as a string
    return random_date.strftime(date_format)

def infer_column_type(column_name):
    """Infer the data type of a column based on its name"""
    column_lower = column_name.lower()

    # Date-related columns
    date_patterns = ['date', 'created', 'updated', 'timestamp', 'time', 'birthday', 'dob', 'birth']
    if any(pattern in column_lower for pattern in date_patterns):
        return 'date'

    # Numeric columns
    numeric_patterns = ['id', 'num', 'count', 'amount', 'price', 'qty', 'quantity', 'age', 'year', 'month', 'day', 'hour', 'minute', 'second']
    if any(pattern in column_lower for pattern in numeric_patterns) or column_lower.endswith('_id'):
        return 'number'

    # Email columns
    if 'email' in column_lower:
        return 'email'

    # Name columns
    name_patterns = ['name', 'first', 'last', 'user', 'customer', 'employee', 'author']
    if any(pattern in column_lower for pattern in name_patterns):
        return 'name'

    # Address columns
    address_patterns = ['address', 'street', 'city', 'state', 'country', 'zip', 'postal']
    if any(pattern in column_lower for pattern in address_patterns):
        return 'address'

    # Phone columns
    if any(pattern in column_lower for pattern in ['phone', 'mobile', 'cell', 'fax']):
        return 'phone'

    # Default to string
    return 'string'

def generate_random_value(column_name, data_type=None, **kwargs):
    """Generate a random value for a column based on its name and type"""
    # If data type is not specified, infer it from the column name
    if not data_type:
        data_type = infer_column_type(column_name)

    # Generate value based on data type
    if data_type == 'date':
        return generate_random_date(
            start_date=kwargs.get('start_date'),
            end_date=kwargs.get('end_date'),
            date_format=kwargs.get('date_format', '%Y-%m-%d')
        )
    elif data_type == 'number':
        return generate_random_number(
            min_val=kwargs.get('min_val', 0),
            max_val=kwargs.get('max_val', 1000),
            decimal_places=kwargs.get('decimal_places', 0)
        )
    elif data_type == 'email':
        return fake.email()
    elif data_type == 'name':
        return fake.name()
    elif data_type == 'address':
        return fake.address().replace('\n', ', ')
    elif data_type == 'phone':
        return fake.phone_number()
    else:  # string or any other type
        return generate_random_string(
            length=kwargs.get('length', 10),
            pattern=kwargs.get('pattern')
        )

def generate_random_data(columns, num_records=10, **kwargs):
    """Generate a DataFrame with random data for the specified columns"""
    data = {}

    for column in columns:
        # Get column-specific parameters if provided
        column_params = kwargs.get(column, {})

        # Generate random values for the column
        data[column] = [
            generate_random_value(
                column,
                data_type=column_params.get('type'),
                length=column_params.get('length', 10),
                pattern=column_params.get('pattern'),
                min_val=column_params.get('min_val', 0),
                max_val=column_params.get('max_val', 1000),
                decimal_places=column_params.get('decimal_places', 0),
                start_date=column_params.get('start_date'),
                end_date=column_params.get('end_date'),
                date_format=column_params.get('date_format', '%Y-%m-%d')
            ) for _ in range(num_records)
        ]

    return pd.DataFrame(data)

def get_table_schema_from_db(connection_url, table_name):
    """Get the schema of a table from the database"""
    try:
        # Create a temporary connection to the database
        from sqlalchemy import create_engine
        engine = create_engine(connection_url)

        # Get the table schema
        query = f"SELECT * FROM {table_name} LIMIT 0"
        df = pd.read_sql(query, engine)

        # Return the column names
        return list(df.columns)
    except Exception as e:
        print(f"Error getting table schema: {str(e)}")
        return []
