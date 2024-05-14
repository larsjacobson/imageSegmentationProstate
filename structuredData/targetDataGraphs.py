import pymongo
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://larsjacobson:4Mf%239x8E@eojgroup.6ifxr.mongodb.net/?retryWrites=true&w=majority")
db = client["prostateSegmentation_MRI"]
collection = db["targetData"]

# Query data from the MongoDB collection
data = list(collection.find({}))

# Extract relevant fields from the data (modify these fields as per your data)
x_values = [entry['Patient ID'] for entry in data]
y_values = [entry['UCLA Score (Similar to PIRADS v2)'] for entry in data]

# Create a line plot
plt.figure(figsize=(10, 6))
sns.palette = sns.color_palette("dark")
plt.plot(x_values, y_values, marker='o', linestyle='-', color='green')
plt.xlabel('Patient ID')
plt.ylabel('UCLA Score (Similar to PIRADS v2)')
plt.title('UCLA Score (Similar to PIRADS v2) per Patient ID')
plt.grid(True)

# Show the line plot
plt.show()

# Extract relevant fields for a bar plot (modify these fields as per your data)
categories = [entry['UCLA Score (Similar to PIRADS v2)'] for entry in data]
values = [entry['ROI Volume (cc)'] for entry in data]

# Create a bar plot
plt.figure(figsize=(10, 6))
sns.palette = sns.color_palette("dark")
sns.barplot(x=categories, y=values, color='green')
plt.xlabel('UCLA Score (Similar to PIRADS v2)')
plt.ylabel('ROI Volume (cc)')
plt.title('ROI Volume (cc) per UCLA Score (Similar to PIRADS v2)')
#plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the bar plot
plt.show()
