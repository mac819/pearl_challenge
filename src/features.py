from pathlib import Path

from loguru import logger
from tqdm import tqdm
import typer

from src.config import PROCESSED_DATA_DIR
import pickle
import pandas as pd


nos_loan_mapping = {
    'zero_nos_active_loan': 0,
    'one_nos_active_loan': 1,
    'low_nos_active_loan': 2,
    'mid_nos_active_loan': 3,
    'high_nos_active_load': 4
}

# ownership_map = {
#     'Owned': 0,
#     'Parental': 1,
#     'Rented': 2
# }

# address_type_encoding_map = {
#     'Permanent Address': 0,
#     'Both Addresses': 1,
#     'Current Address': 2
# }

village_non_agri_income_label_mapping = {
    'order0_median_income': 0,
    'order1_median_income': 1,
    'order2_median_income': 2,
    'order3_median_income': 3,
    'order4_median_income': 4
}

village_agri_lan_bin_map = {
    'order0_vill_agri_land_bin': 0,
    'order1_vill_agri_land_bin': 1,
    'order2_vill_agri_land_bin': 2,
    'order3_vill_agri_land_bin': 3
}

district_non_agri_bin_map = {
    'order0_median_income_bin': 0,
    'order1_median_income_bin': 1,
    'order2_median_income_bin': 2,
    'order3_median_income_bin': 3,
    'order4_median_income_bin': 4
}


def create_nos_active_loan_bins(pearl_train_df, is_train=True):

    def loan_bins(x):
        """
        Intervals: 
        [0] --> 0
        [1] --> 1
        [2, 3] --> 2
        [4] --> 3
        [5, 64] --> 4
        """

        if x['no_of_active_loan_in_bureau'] == 0:
            x['active_loan_bins'] = 0
        elif x['no_of_active_loan_in_bureau'] == 1:
            x['active_loan_bins'] = 1
        elif (x['no_of_active_loan_in_bureau'] >= 2) and (x['no_of_active_loan_in_bureau'] <= 3):
            x['active_loan_bins'] = 2
        elif x['no_of_active_loan_in_bureau'] == 4:
            x['active_loan_bins'] = 3
        else:
            x['active_loan_bins'] = 4
        return x

    if is_train:
        zero_active_loan_pearl_train_df = pearl_train_df[pearl_train_df['no_of_active_loan_in_bureau']==0].reset_index(drop=True)
        zero_active_loan_pearl_train_df = zero_active_loan_pearl_train_df[['FarmerID']].drop_duplicates()
        zero_active_loan_pearl_train_df['active_loan_bins'] = 'zero_nos_active_loan'
        zero_active_loan_pearl_train_df = zero_active_loan_pearl_train_df[['FarmerID', 'active_loan_bins']]
        
        one_active_loan_pearl_train_df = pearl_train_df[pearl_train_df['no_of_active_loan_in_bureau']==1].reset_index(drop=True)
        one_active_loan_pearl_train_df = one_active_loan_pearl_train_df[['FarmerID']].drop_duplicates()
        one_active_loan_pearl_train_df['active_loan_bins'] = 'one_nos_active_loan'
        one_active_loan_pearl_train_df = one_active_loan_pearl_train_df[['FarmerID', 'active_loan_bins']]
        
        remaining_active_load_pearl_train_df = pearl_train_df[~pearl_train_df['no_of_active_loan_in_bureau'].isin([0, 1])].reset_index(drop=True)
        active_loan_bins = ['low_nos_active_loan', 'mid_nos_active_loan', 'high_nos_active_load']
        remaining_active_load_pearl_train_df['active_loan_bins'] = pd.qcut(
            remaining_active_load_pearl_train_df['no_of_active_loan_in_bureau'], 
            q=3, 
            labels=active_loan_bins
        )
        remaining_active_load_pearl_train_df = remaining_active_load_pearl_train_df[['FarmerID', 'active_loan_bins']]
        loan_bins_total_df = pd.concat(
            [zero_active_loan_pearl_train_df, one_active_loan_pearl_train_df, remaining_active_load_pearl_train_df], 
            axis=0
        )

        pearl_train_df = pearl_train_df.merge(
            right=loan_bins_total_df,
            on='FarmerID'
        )
        
        pearl_train_df['active_loan_bins'] = pearl_train_df['active_loan_bins'].map(nos_loan_mapping)
    else:
        pearl_train_df = pearl_train_df.apply(lambda x: loan_bins(x), axis=1)
    return pearl_train_df, ['FarmerID', 'active_loan_bins']



# def ownership_encoding(data):
#     data['enc_ownership'] = data['Ownership'].map(ownership_map)
#     feature_columns = ['FarmerID', 'enc_ownership']
#     return data, feature_columns

# def addresstype_encoding(data):
#     data['address_type'] = data['Address type'].map(address_type_encoding_map)
#     return data, ['FarmerID', 'address_type']

def village_clubbing(data, exp_name, is_train=True):
    pearl_train_df = data.copy()

    def village_mapping(x):
        if vill_name_mapping.get(x['VILLAGE']):
            x['village_en_name'] = vill_name_mapping[x['VILLAGE']]
        else:
            x['village_en_name'] = 'others'
        return x
    
    if is_train:
        village_count_df = pd.DataFrame(pearl_train_df['VILLAGE'].value_counts()).reset_index()

        # Village to Others Mapping
        village_low_count_df = village_count_df[village_count_df['count'] <= 5].reset_index(drop=True)
        # village_low_count_df['village_encoded_name'] = 'others'
        vill_name_mapping = {}
        low_count_village_list = village_low_count_df['VILLAGE'].unique()
        for vil in low_count_village_list:
            vill_name_mapping[vil] = 'others'

        remaining_village_count_df = village_count_df[village_count_df['count'] > 5].reset_index(drop=True)
        for row in remaining_village_count_df.iterrows():
            vill_name_mapping[row[1]['VILLAGE']] = row[1]['VILLAGE'].lower()

        pearl_train_df['village_en_name'] = pearl_train_df['VILLAGE'].map(vill_name_mapping)

        # Saving village group mapping
        with open(PROCESSED_DATA_DIR / exp_name / 'village_group_map.pkl', 'wb') as f:
            pickle.dump(vill_name_mapping, f)
    else:
        with open(PROCESSED_DATA_DIR / exp_name / 'village_group_map.pkl', 'rb') as f:
            vill_name_mapping = pickle.load(f)
        pearl_train_df = pearl_train_df.apply(lambda x: village_mapping(x), axis=1)

    return pearl_train_df, ['FarmerID', 'village_en_name']


def village_non_agri_income_bins(data, is_train=True):
    """
    Intervals: [(0.499, 100000.0] < (100000.0, 145000.0] < (145000.0, 200000.0] < (200000.0, 967960.0]]
    """

    def test_village_non_agri_income_bins(x):
        if x['non_agriculture_income'] == 0:
            x['village_non_agri_income_bin'] = 0
        elif (x['non_agriculture_income'] > 0.0) and (x['non_agriculture_income'] <= 100000.0):
            x['village_non_agri_income_bin'] = 1
        elif (x['non_agriculture_income'] > 100000.0) and (x['non_agriculture_income'] <= 145000.0):
            x['village_non_agri_income_bin'] = 2
        elif (x['non_agriculture_income'] > 145000.0) and (x['non_agriculture_income'] <= 200000.0):
            x['village_non_agri_income_bin'] = 3
        else:
            x['village_non_agri_income_bin'] = 4
        return x

    pearl_train_df = data.copy()

    if is_train:
        village_non_agri_income_median_df = pearl_train_df.groupby('village_en_name').agg({'non_agriculture_income': 'median'}).reset_index()
        zero_median_non_agri_income_village_df = village_non_agri_income_median_df[village_non_agri_income_median_df['non_agriculture_income']==0].reset_index(drop=True)
        non_zero_median_non_agri_income_village_df = village_non_agri_income_median_df[village_non_agri_income_median_df['non_agriculture_income']!=0].reset_index(drop=True)

        labels = ['order1_median_income', 'order2_median_income', 'order3_median_income', 'order4_median_income']
        non_zero_median_non_agri_income_village_df['village_non_agri_income_bin'] = pd.qcut(
            non_zero_median_non_agri_income_village_df['non_agriculture_income'], 
            q=4, 
            labels=labels
        )
        zero_median_non_agri_income_village_df['village_non_agri_income_bin'] = 'order0_median_income'

        village_non_agri_bin_df = pd.concat([non_zero_median_non_agri_income_village_df, zero_median_non_agri_income_village_df], axis=0)

        pearl_train_df = pearl_train_df.merge(
            right=village_non_agri_bin_df[['village_en_name', 'village_non_agri_income_bin']],
            on='village_en_name'
        )

        pearl_train_df['village_non_agri_income_bin'] = pearl_train_df['village_non_agri_income_bin'].map(village_non_agri_income_label_mapping)
    else:
        pearl_train_df = pearl_train_df.apply(lambda x: test_village_non_agri_income_bins(x), axis=1)

    return pearl_train_df, ['FarmerID', 'village_non_agri_income_bin']


def village_agri_land_bins(data, is_train=True):
    """
    Intervals: [(1.999, 5.25] < (5.25, 9.0] < (9.0, 10.0] < (10.0, 27.0]]
    """

    def test_village_agri_land_bins(x):
        if (x['Total_Land_For_Agriculture'] <= 5.25):
            x['village_total_land_for_agriculture_bin'] = 0
        elif (x['Total_Land_For_Agriculture'] > 5.25) and (x['Total_Land_For_Agriculture'] <= 9.0):
            x['village_total_land_for_agriculture_bin'] = 1
        elif (x['Total_Land_For_Agriculture'] > 9.0) and (x['Total_Land_For_Agriculture'] <= 10.0):
            x['village_total_land_for_agriculture_bin'] = 2
        else:
            x['village_total_land_for_agriculture_bin'] = 3
        return x

    pearl_train_df = data.copy()

    if is_train:
        village_agri_land_df = pearl_train_df.groupby('village_en_name').agg({'Total_Land_For_Agriculture': 'median'}).reset_index()

        village_agri_land_df['total_vill_land_for_agriculture_bin'] = pd.qcut(
            village_agri_land_df['Total_Land_For_Agriculture'],
            q=4,
            labels=['order0_vill_agri_land_bin', 'order1_vill_agri_land_bin', 'order2_vill_agri_land_bin', 'order3_vill_agri_land_bin']
        )

        pearl_train_df = pearl_train_df.merge(
            right=village_agri_land_df[['village_en_name', 'total_vill_land_for_agriculture_bin']],
            on='village_en_name'
        )

        pearl_train_df['village_total_land_for_agriculture_bin'] = pearl_train_df['total_vill_land_for_agriculture_bin'].map(village_agri_lan_bin_map)
    else:
        pearl_train_df = pearl_train_df.apply(lambda x: test_village_agri_land_bins(x), axis=1)
    return pearl_train_df, ['FarmerID', 'village_total_land_for_agriculture_bin']


def district_clubbing(data, exp_name, is_train=True):

    def district_mapping(x):
        if district_name_mapping.get(x['DISTRICT']):
            x['district_en_name'] = district_name_mapping[x['DISTRICT']]
        else:
            x['district_en_name'] = 'others'
        return x
    
    pearl_train_df = data.copy()
    if is_train:
        district_value_count_df = pd.DataFrame(pearl_train_df['DISTRICT'].value_counts()).reset_index()

        district_low_count_df = district_value_count_df[district_value_count_df['count'] <= 5].reset_index(drop=True)
        district_name_mapping = {}
        low_count_district_list = district_low_count_df['DISTRICT'].unique()
        for dist in low_count_district_list:
            district_name_mapping[dist] = 'others'

        remaining_district_count_df = district_value_count_df[district_value_count_df['count'] > 5].reset_index(drop=True)
        for row in remaining_district_count_df.iterrows():
            district_name_mapping[row[1]['DISTRICT']] = row[1]['DISTRICT'].lower()
        pearl_train_df['district_en_name'] = pearl_train_df['DISTRICT'].map(district_name_mapping)

        # Saving village group mapping
        with open(PROCESSED_DATA_DIR / exp_name / 'district_group_map.pkl', 'wb') as f:
            pickle.dump(district_name_mapping, f)
    else:
        with open(PROCESSED_DATA_DIR / exp_name / 'district_group_map.pkl', 'rb') as f:
            district_name_mapping = pickle.load(f)
        pearl_train_df = pearl_train_df.apply(lambda x: district_mapping(x), axis=1)

    return pearl_train_df, ['FarmerID', 'district_en_name']


def district_non_ari_income_bins(data, is_train=True):
    """
    Intervals: [(0.999, 123750.0] < (123750.0, 200000.0] < (200000.0, 302000.0] < (302000.0, 740000.0]]
    """
    def test_district_non_agri_income_bin(x):
        if x['non_agriculture_income'] == 0:
            x['district_non_agri_bin'] = 0
        elif (x['non_agriculture_income'] > 0.999) and (x['non_agriculture_income'] <= 123750.0):
            x['district_non_agri_bin'] = 1
        elif (x['non_agriculture_income'] > 123750.0) and (x['non_agriculture_income'] <= 200000.0):
            x['district_non_agri_bin'] = 2
        elif (x['non_agriculture_income'] > 200000.0) and (x['non_agriculture_income'] <= 302000.0):
            x['district_non_agri_bin'] = 3
        else:
            x['district_non_agri_bin'] = 4
        return x

    pearl_train_df = data.copy()

    if is_train:
        district_non_agri_income_median_df = pearl_train_df.groupby('district_en_name').agg({'non_agriculture_income': 'median'}).reset_index()

        zero_district_non_agri_median_income_df = district_non_agri_income_median_df[
            district_non_agri_income_median_df['non_agriculture_income']==0
        ].reset_index(drop=True)
        non_zero_district_non_agri_median_income_df = district_non_agri_income_median_df[
            district_non_agri_income_median_df['non_agriculture_income']!=0
        ].reset_index(drop=True)

        non_zero_district_non_agri_median_income_df['district_non_agri_bin'] = pd.qcut(
            non_zero_district_non_agri_median_income_df['non_agriculture_income'],
            q=4,
            labels=['order1_median_income_bin', 'order2_median_income_bin', 'order3_median_income_bin', 'order4_median_income_bin']
        )
        zero_district_non_agri_median_income_df['district_non_agri_bin'] = 'order0_median_income_bin'
        district_non_agri_income_median_df = pd.concat(
            [non_zero_district_non_agri_median_income_df, zero_district_non_agri_median_income_df], 
            axis=0
        )

        pearl_train_df = pearl_train_df.merge(
            right=district_non_agri_income_median_df[['district_en_name', 'district_non_agri_bin']],
            on='district_en_name'
        )

        pearl_train_df['district_non_agri_bin'] = pearl_train_df['district_non_agri_bin'].map(district_non_agri_bin_map)
    else:
        pearl_train_df = pearl_train_df.apply(lambda x: test_district_non_agri_income_bin(x), axis=1)
    return pearl_train_df, ['FarmerID', 'district_non_agri_bin']


def district_agri_land_bins(data, is_train=True):
    """
    Intervals: [(1.999, 5.25] < (5.25, 9.0] < (9.0, 10.0] < (10.0, 27.0]]
    """

    def test_district_agri_land_bins(x):
        if x['Total_Land_For_Agriculture'] <= 5.25:
            x['district_total_land_for_agriculture_bin'] = 0
        elif (x['Total_Land_For_Agriculture'] > 5.25) and (x['Total_Land_For_Agriculture'] <= 9.0):
            x['district_total_land_for_agriculture_bin'] = 1
        elif (x['Total_Land_For_Agriculture'] > 9.0) and (x['Total_Land_For_Agriculture'] <= 10.0):
            x['district_total_land_for_agriculture_bin'] = 2
        else:
            x['district_total_land_for_agriculture_bin'] = 3
        return x
    
    pearl_train_df = data.copy()

    if is_train:
        district_agri_land_df = pearl_train_df.groupby('district_en_name').agg({'Total_Land_For_Agriculture': 'median'}).reset_index()

        district_agri_land_df['district_total_land_for_agriculture_bin'] = pd.qcut(
            district_agri_land_df['Total_Land_For_Agriculture'],
            q=4,
            labels=['order0_dist_agri_land_bin', 'order1_dist_agri_land_bin', 'order2_dist_agri_land_bin', 'order3_dist_agri_land_bin']
        )

        pearl_train_df = pearl_train_df.merge(
            right=district_agri_land_df[['district_en_name', 'district_total_land_for_agriculture_bin']],
            on='district_en_name'
        )

        district_land_bin_map = {
            'order0_dist_agri_land_bin': 0,
            'order1_dist_agri_land_bin': 1,
            'order2_dist_agri_land_bin': 2,
            'order3_dist_agri_land_bin': 3
        }

        pearl_train_df['district_total_land_for_agriculture_bin'] = pearl_train_df['district_total_land_for_agriculture_bin'].map(district_land_bin_map)
    else:
        pearl_train_df = pearl_train_df.apply(lambda x: test_district_agri_land_bins(x), axis=1)
    return pearl_train_df, ['FarmerID', 'district_total_land_for_agriculture_bin']
    