import os
import random
import shutil
from sklearn.model_selection import train_test_split


input_dir = './data/conllu'
output_dir = './data/splits'

train_dir = os.path.join(output_dir,'train')
test_dir = os.path.join(output_dir,'test')
val_dir = os.path.join(output_dir,'validation')

train_ratio = 0.7       #70% training
test_val_ratio = 0.5    #15% testing & 15% validation

random_seed = 42

def split_dir():
    """ Recreates the output directories for nice splitting """
    
    for dir in [train_dir, test_dir, val_dir]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir, exist_ok=True)
    print(f"Output dir created/cleaned in: {output_dir}")

def split_and_copy():
    """ reads the .conllu files and does a random split, copying them into respective folder"""
    files = [f for f in os.listdir(input_dir) if f.endswith('.conllu')]

    train_files, temp_files = train_test_split(files, test_size= (1.0-train_ratio), 
                                               random_state=random_seed)
    val_files, test_files = train_test_split(temp_files, test_size= (1.0-test_val_ratio), 
                                             random_state=random_seed)
    
    split_dir()

    print("\nStarting file copy")

    def copy_set(file_list, new_dir, name):

        print(f"\nCopying {len(file_list)} files to {name} set ({new_dir}):")

        for filename in file_list:
            source_path = os.path.join(input_dir, filename)
            new_path = os.path.join(new_dir, filename)
            shutil.copy2(source_path,new_path)

        print(f"Finished copying {len(file_list)} files for {name}")

    copy_set(train_files, train_dir, "Training")
    copy_set(test_files, test_dir, "Testing")
    copy_set(val_files, val_dir, "Validation")

    print("\n Data splitting and copying completed")

if __name__ == "__main__":
    split_and_copy()