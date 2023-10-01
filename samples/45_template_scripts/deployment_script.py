"""
Purpose:
Data preparation
	- download
	- cleansing
	- split
	- save
- Configure estimator
- Start training by calling 'estimator.fit()'
- Configure predictor
- Start prediction by calling predictor
- Delete predictor.endpoint

"""

# region IMPORT packages

# my own little helpers
import common_dataframe_utils
import common_tokenizer_utils
import common_image_utils
import common_file_utils

import sagemaker

import json
import os

import pandas as pd
import numpy as np

# Tools for downloading images and corresponding annotations from Google's OpenImages dataset.
from openimages.download import _download_images_by_id

# endregion


# region VARIABLE definition
prefix = "image2caption"

max_images = 100                                                # number of images to be downloaded
validation_share = 0.1                                          # Share of validation images on total

file_train_validation_csv = "image_data.csv"                    # list of files selected for training and validation
file_json_dic = "open_images_train_v6_captions.jsonl"           # dictionary with images and their captions

dir_training_images = \
    "data/20_cleansed/google_images/training"                   # directory for training images
dir_validation_images = \
    "data/20_cleansed/google_images/validation"                 # directory for validation images
dir_helper = "/Users/swmoeller/python/prj_image_captioning_e2e/data/00_helpers"                                  # location for helper files such as json_input, etc.

# conversion into directories


# endregion

# region SAGEMAKER definition
sagemaker_session = sagemaker.Session()
bucket = sagemaker_session.default_bucket()

# role = sagemaker.get_execution_role()
role_input = "arn:aws:iam::125232469585:role/service-role/AmazonSageMaker-ExecutionRole-20220909T131699"

# endregion


# region FUNCTION definition

def download_images(IN_max_images: int,
                    IN_path_json_dic: str,
                    IN_validation_share: float,
                    IN_path_train_validation_csv: str,
                    IN_dir_training_images: str,
                    IN_dir_validation_images: str):

    """
    Generates a training and validation dataset using a json file containing image name, caption and other information.

    1. Loads the json file
    2. Randomly picks the requested number of images from the json file and safes the information about each image as
        dataframe
    3. Inserts another column into the dataframe and classifies each image as training or validation image in this column
    4. Processes the dataframe and downloads the images into the two folders for training and validation

    :param IN_image_json_file: full path to json file including the dictionary with image names, captions, etc.
    :type IN_image_json_file: str
    :param IN_max_amount_of_images: Maximum amount of images to be downloaded
    :type IN_max_amount_of_images: int
    :param IN_validation_share: Percentage (in float) of images kept as validation images (this not included in
    training)
    :type IN_validation_share: float
    :return: pandas dataframe containing the randomly selected images with their captions, etc.
    :rtype: dataframe
    """
    

    
    
    # Fetch the dataset from the Open Images dataset
    with open(IN_path_json_dic, 'r') as json_file:
        json_list = json_file.read().split('\n')
    np.random.shuffle(json_list)  # avoid always downloading the same images
    print(f"[Info] JSON file {IN_path_json_dic} loaded and shuffled!")
    
    # Generate a list of dataframes containing image name, caption and other information as column names and listing the
    # images and their details in each line
    data_df_list = []  # list containing later the single dataframes
    for index_no, json_str in enumerate(json_list):
        if index_no == IN_max_images:
            break
        try:
            result = json.loads(json_str)
            
            """
            Create a DataFrame from a dictionary "result" with orient='index' argument to use the keys of the dictionary
            as the column names of the DataFrame.
            The .T attribute is then used to transpose the resulting DataFrame. The transpose operation switches the
            rows and columns of the DataFrame.
            """
            x = pd.DataFrame.from_dict(result, orient='index').T
            data_df_list.append(x)
        except:
            pass
    
    # Generate one dataframe out of the list of single datafranes and split the dataframe (data) into training and
    # validation data (i.e. mark the images as validation or training image)
    np.random.seed(10)
    data_df = pd.concat(data_df_list)
    # generate a list of all images with one image and its caption in one line plus an indication (true, false),
    # if used as training image
    data_df['train'] = np.random.choice([True, False],
                                        size=len(data_df),
                                        p=[1 - IN_validation_share, IN_validation_share])
    
    # save the dataframe as csv file (without index)
    data_df.to_csv(IN_path_train_validation_csv, index=False)
    
    # Download all images marked in column "train" as "True" corresponding to the image IDs fetched from the JSON
    subset_imageIds = common_dataframe_utils.extract_values_from_column(
        IN_pandas_dataframe=data_df,
        IN_filter_column_name="train",
        IN_filter_criteria=True,
        OUT_column_name="image_id"
    )
    _download_images_by_id(subset_imageIds, 'train', IN_dir_training_images)
    print(f"[INFO] I downloaded {len(subset_imageIds)} images")
    
    # Download all images marked in column "train" as "False" (i.e. validation images) corresponding to the image IDs
    # fetched from the JSON
    subset_imageIds = common_dataframe_utils.extract_values_from_column(
        IN_pandas_dataframe=data_df,
        IN_filter_column_name="train",
        IN_filter_criteria=False,
        OUT_column_name="image_id"
    )
    _download_images_by_id(subset_imageIds, 'train', IN_dir_validation_images)
    
    return data_df


# endregion

# region MAIN

df_images = download_images(IN_max_images=max_images,
                            IN_path_json_dic=os.path.join(dir_helper,file_json_dic),
                            IN_validation_share=validation_share,
                            IN_path_train_validation_csv=os.path.join(dir_helper,file_train_validation_csv),
                            IN_dir_training_images=dir_training_images,
                            IN_dir_validation_images=dir_validation_images)

print(df_images)

# endregion
