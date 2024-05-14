
import pandas as pd
from pymongo import MongoClient

# MongoDB connection settings
mongo_uri = "mongodb+srv://larsjacobson:4Mf%239x8E@eojgroup.6ifxr.mongodb.net/?retryWrites=true&w=majority"
database_name = "prostateSegmentation_MRI"
collection_name = "targetData" #"biopsyData"

# Connect to MongoDB
client = MongoClient(mongo_uri)
db = client[database_name]
collection = db[collection_name]

# Read the Excel file
excel_file = "Target Data_2019-12-05.xlsx"  #"TCIA Biopsy Data_2020-07-14"
df = pd.read_excel(excel_file)

# Convert each row to a dynamic JSON document and insert it into MongoDB
for _, row in df.iterrows():
    # Create a dynamic JSON document using column names as keys
    json_document = {col: value for col, value in row.items()}
    collection.insert_one(json_document)

# Close the MongoDB connection
client.close()
