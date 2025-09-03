import os
from dotenv import load_dotenv
from azure.identity import InteractiveBrowserCredential
import sqlalchemy as sa
import urllib
import struct
from itertools import chain, repeat

load_dotenv()

# Hardcoded resource URL for Azure SQL scope
RESOURCE_URL = "https://database.windows.net/.default"

# Interactive login (will open browser if needed)
credential = InteractiveBrowserCredential()

token_object = credential.get_token(RESOURCE_URL)
auth_token = token_object.token

sql_endpoint = os.getenv("SQL_ENDPOINT")
database = os.getenv("LAKEHOUSE_DB")
connection_string = f"Driver={{ODBC Driver 18 for SQL Server}};Server={sql_endpoint},1433;Database={database};Encrypt=Yes;TrustServerCertificate=No"
params = urllib.parse.quote(connection_string)

# Convert token to Windows byte string format
token_as_bytes = bytes(auth_token, "UTF-8")
encoded_bytes = bytes(chain.from_iterable(zip(token_as_bytes, repeat(0))))
token_bytes = struct.pack("<i", len(encoded_bytes)) + encoded_bytes
attrs_before = {1256: token_bytes}

# Establish SQLAlchemy engine
engine = sa.create_engine(f"mssql+pyodbc:///?odbc_connect={params}", connect_args={'attrs_before': attrs_before})

from sqlalchemy import text

with engine.connect() as conn:
    # list available tables in the database
    result = conn.execute(text("SELECT TABLE_SCHEMA, TABLE_NAME FROM INFORMATION_SCHEMA.TABLES"))
    for row in result:
        print(row)
