# import os
from datasets import load_dataset

def build_index_for_fineweb(dataset_name, sample, split):
    fineweb = load_dataset(dataset_name, sample, split=split, num_proc=12, revision="042ac03070484d97ab32e6899e1c2b571b2e9c38") # cache_dir="/Volumes/RAID1/hf-datasets")
    fineweb.add_elasticsearch_index(column="text", host="localhost", port="9200", es_index_name=f"hf_{dataset_name.replace('/', '_')}_{sample_name}_{split}_text".lower())

if __name__ == '__main__':
    dataset_name = 'HuggingFaceFW/fineweb'
    sample_name = 'sample-350BT'
    split = 'train'

    build_index_for_fineweb(dataset_name, sample_name, split)
