import numpy as np
import pandas as pd
from faker import Faker
# will create three types of datasets - users, datasets, downloads

# users DB

fake = Faker()

types_of_occupation = ['student', 'industrialist', 'professor', 'government_employee']

users_df = pd.DataFrame([
    {   
        "user_id": str(x)  ,
        "name" : fake.name(), 
        "occupation": types_of_occupation[np.random.randint(low = 0, high = 4, size=(1))[0]]
    }

    for x in range(1000)
])

#datasets DB

types_of_datasets = ['agriculture', 'water', 'transport', 'forestry', 'tourism', 'urban', 'rural']
types_of_usefullness = ['teaching', 'research', 'industry', 'survey']
indian_states = ["Andhra Pradesh",
               "Arunachal Pradesh ",
               "Assam","Bihar",
               "Chhattisgarh",
               "Goa",
               "Gujarat",
               "Haryana",
               "Himachal Pradesh",
               "Jammu and Kashmir",
               "Jharkhand",
               "Karnataka",
               "Kerala",
               "Madhya Pradesh",
               "Maharashtra",
               "Manipur",
               "Meghalaya",
               "Mizoram",
               "Nagaland",
               "Odisha",
               "Punjab",
               "Rajasthan",
               "Sikkim",
               "Tamil Nadu",
               "Telangana",
               "Tripura",
               "Uttar Pradesh",
               "Uttarakhand",
               "West Bengal",
               "Andaman and Nicobar Islands",
               "Chandigarh",
               "Dadra and Nagar Haveli",
               "Daman and Diu",
               "Lakshadweep",
               "National Capital Territory of Delhi",
               "Puducherry"]

datasets_df = pd.DataFrame([
    {
        "dataset_id": str(x) ,
        "type_of_dataset": types_of_datasets[np.random.randint(low = 0, high = len(types_of_datasets), size=(1))[0]] ,
        "state": indian_states[np.random.randint(low = 0, high = len(indian_states), size=(1))[0]],
        "useful_for": types_of_usefullness[np.random.randint(low = 0, high = len(types_of_usefullness), size=(1))[0]]
    }
    for x in range(500)
])

titles = []
for i in range(datasets_df.shape[0]):
    title = f"{datasets_df['type_of_dataset'].iloc[i]} dataset of {datasets_df['state'].iloc[i]} for {datasets_df['useful_for'].iloc[i]}"
    titles.append(title)

datasets_df['title'] = titles

# downloads DB

downloads_df = pd.DataFrame([{
    "download_id": str(x) ,
    "user_id": str(np.random.randint(low = 0, high = len(users_df), size = 1)[0]) ,
    "dataset_id": str(np.random.randint(low = 0, high = len(datasets_df), size = 1)[0]) 
}
    for x in range(100_000)

])
download_titles = []
for i in range(downloads_df.shape[0]):
    download_titles.append(datasets_df['title'][int(downloads_df['dataset_id'].iloc[i])])

downloads_df['title'] = download_titles






users_df.to_csv('./data/users_db.csv')
datasets_df.to_csv('./data/datasets_db.csv')
downloads_df.to_csv('./data/downloads_db.csv')




