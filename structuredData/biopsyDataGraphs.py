import pymongo
import matplotlib.pyplot as plt
import seaborn as sns

# Connect to MongoDB
client = pymongo.MongoClient("mongodb+srv://larsjacobson:4Mf%239x8E@eojgroup.6ifxr.mongodb.net/?retryWrites=true&w=majority")
db = client["prostateSegmentation_MRI"]
collection = db["biopsyData"]

# Query data from the MongoDB collection
data = list(collection.find({}))

# Extract relevant fields from the data (modify these fields as per your data)
x_values = [entry['Patient Number'] for entry in data]
y_values = [entry['PSA (ng/mL)'] for entry in data]

# Create a line plot
plt.figure(figsize=(20, 10))
plt.plot(x_values, y_values, marker='o', linestyle='-')
plt.xlabel('Patient Number')
plt.ylabel('PSA (ng/mL)')
plt.title('PSA (ng/mL) per Patient ID')
plt.grid(True)

# Show the line plot
plt.show()

# Extract relevant fields for a bar plot (modify these fields as per your data)
categories = [entry['Core Label'] for entry in data]
values = [entry['PSA (ng/mL)'] for entry in data]

# Create a bar plot
plt.figure(figsize=(20, 10))
sns.barplot(x=categories, y=values)
plt.xlabel('Core Label')
plt.ylabel('PSA (ng/mL)')
plt.title('PSA (ng/mL) per Core Label')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Show the bar plot
plt.show()
