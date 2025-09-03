from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv("SQL_ENDPOINT"))  # Should print your hostname, not None