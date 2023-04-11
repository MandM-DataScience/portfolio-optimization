from configparser import ConfigParser

import pymongo
from pymongo import MongoClient
import os
import json

from portfolio_optimization import PORTFOLIO_BASE_DIR

db_name = 'portfolio'

def get_mongodb_client():

    parser = ConfigParser()
    _ = parser.read(os.path.join(PORTFOLIO_BASE_DIR,"credentials.cfg"))
    username = parser.get("mongo_db", "username")
    password = parser.get("mongo_db", "password")

    LOCAL_CONNECTION = "mongodb://localhost:27017"
    ATLAS_CONNECTION = f"mongodb+srv://{username}:{password}@cluster0.3dxfmjo.mongodb.net/?" \
                       f"retryWrites=true&w=majority"

    # print(ATLAS_CONNECTION)

    # Provide the mongodb atlas url to connect python to mongodb using pymongo
    connection_string = ATLAS_CONNECTION
    # Create a connection using MongoClient. You can import MongoClient or use pymongo.MongoClient
    client = MongoClient(connection_string)
    # Create the database for our example (we will use the same database throughout the tutorial
    return client

def get_database(db_name):
    return get_mongodb_client()[db_name]

def get_collection(collection_name):
    db = get_database(db_name)
    return db[collection_name]

def get_datasets_collection():
    collection_name = 'oecd_datasets'
    return get_collection(collection_name)

def get_fred_categories_collection():
    collection_name = 'fred_categories'
    return get_collection(collection_name)

def get_fred_series_collection():
    collection_name = 'fred_series'
    return get_collection(collection_name)

def get_fred_datasets_collection():
    collection_name = 'fred_datasets'
    return get_collection(collection_name)

def get_yf_target_datasets_collection():
    collection_name = 'yf_target_datasets'
    return get_collection(collection_name)

def get_investing_target_datasets_collection():
    collection_name = 'investing_target_datasets'
    return get_collection(collection_name)

def get_feature_selection_collection():
    collection_name = 'feature_selection'
    return get_collection(collection_name)

def upload_dataset(data):
    datasets = get_datasets_collection()
    datasets.insert_one(data)

def upload_fred_category(data):
    fred_categories = get_fred_categories_collection()
    fred_categories.insert_one(data)

def upload_feature_selection(data):
    feature_selection = get_feature_selection_collection()
    feature_selection.insert_one(data)

def upload_fred_series(data):
    fred_series = get_fred_series_collection()
    fred_series.insert_one(data)

def upload_fred_datasets(data):
    fred_datasets = get_fred_datasets_collection()
    fred_datasets.insert_one(data)

def upload_fred_series_many(data):
    fred_series = get_fred_series_collection()
    fred_series.insert_many(data)

def upload_yf_target_datasets(data):
    yf_target_datasets = get_yf_target_datasets_collection()
    yf_target_datasets.insert_one(data)

def upload_investing_target_datasets(data):
    investing_target_datasets = get_investing_target_datasets_collection()
    investing_target_datasets.insert_one(data)

def exist_dataset(dataset_id=None):
    datasets = get_datasets_collection()
    result = datasets.count_documents({"metadata.dataset_id": dataset_id})
    return result > 0

def get_file_size(file_name):
    file_stats = os.stat(file_name)
    print(f'File Size in Bytes is {file_stats.st_size}')
    return file_stats.st_size

def get_dict_size(data):
    import sys
    print("The size of the dictionary is {} bytes".format(sys.getsizeof(data)))
    return sys.getsizeof(data)

def insert_datasets():
    with open('oecd_api_dataset_ids.json', 'r', encoding='utf-8') as f:
        datasets = json.loads(f.read())

    dir_list = os.listdir('./datasets')
    for file_name in dir_list:
        file_path = f"./datasets/{file_name}"
        dataset_id = file_name.replace('.json','')
        if not exist_dataset(dataset_id):
            metadata = None
            for d in datasets:
                if 'dataset_id' in d and d['dataset_id'] == dataset_id:
                    if 'too_large_for_mongodb' in d and d['too_large_for_mongodb']:
                        continue
                    metadata = d
            if metadata is not None:
                print(f"PREPARING: {dataset_id}")

                # return
                with open(file_path, 'r', encoding='utf-8') as f:
                    dataset_file = json.loads(f.read())
                data = {
                    "metadata": metadata,
                    "dataset": dataset_file
                }
                print(f"UPLOAD {dataset_id}")
                try:
                    upload_dataset(data) # max pymongo.errors.DocumentTooLarge: BSON document too large (51200225 bytes) - the connected server supports BSON document sizes up to 16793598 bytes
                except pymongo.errors.DocumentTooLarge as e:
                    print(e)
                    metadata['too_large_for_mongodb'] = True
                    with open('oecd_api_dataset_ids.json', 'w', encoding='utf-8') as f:
                         f.write(json.dumps(datasets))
    # print(dir_list)
    return

# def append_document(collection_name, data):
#     collection = get_collection(collection_name)
#     collection.update_one(
#         {'_id': data["_id"]},
#         {'$push': {'data': {"$each": data['data']}}})

def upsert_document(collection_name, data):
    collection = get_collection(collection_name)
    collection.replace_one({"_id":data["_id"]}, data, upsert=True)

def insert_document(collection_name, data):
    collection = get_collection(collection_name)
    collection.insert_one(data)

def get_document(collection_name, document_id):
    collection = get_collection(collection_name)
    return collection.find({"_id": document_id}).next()

def get_collection_documents(collection_name):
    collection = get_collection(collection_name)
    return collection.find({})