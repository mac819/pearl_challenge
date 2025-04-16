import os
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder


from src.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, soil_type_mapping, ecological_subzone_mapping
from src.features import (
    create_nos_active_loan_bins,
    # ownership_encoding,
    # addresstype_encoding,
    village_clubbing,
    village_non_agri_income_bins,
    village_agri_land_bins,
    district_clubbing,
    district_non_ari_income_bins,
    district_agri_land_bins
)


def load_data(train=True):
    pearl_data_xls = pd.read_excel(
        str(RAW_DATA_DIR / 'Pearl Challenge data with dictionary_For_Share_v4.xlsx'), 
        sheet_name=['TrainData', 'TestData']
    )

    if train:
        data = pearl_data_xls['TrainData']
        data = data.drop_duplicates(subset=['FarmerID'])
    else:
        data = pearl_data_xls['TestData']

    data.rename(
        columns={
            ' Night light index': 'Night light index',
            ' Village category based on socio-economic parameters (Good, Average, Poor)': 'Village category based on socio-economic parameters (Good, Average, Poor)',
            ' Village score based on socio-economic parameters (Non normalised)': 'Village score based on socio-economic parameters (Non normalised)',
            ' Village score based on socio-economic parameters (0 to 100)': 'Village score based on socio-economic parameters (0 to 100)',
            ' Land Holding Index source (Total Agri Area/ no of people)': 'Land Holding Index source (Total Agri Area/ no of people)',
            ' Road density (Km/ SqKm)': 'Road density (Km/ SqKm)'
        }, 
        inplace=True)
    
    if train:
        data = data.dropna(
            subset=[
                'Perc_of_house_with_6plus_room',
                'Women_15_19_Mothers_or_Pregnant_at_time_of_survey',
                'perc_of_pop_living_in_hh_electricity',
                'perc_Households_with_Pucca_House_That_Has_More_Than_3_Rooms',
                'mat_roof_Metal_GI_Asbestos_sheets',
                'perc_of_Wall_material_with_Burnt_brick',
                'Households_with_improved_Sanitation_Facility',
                'perc_Households_do_not_have_KCC_With_The_Credit_Limit_Of_50k'
            ]
        )
    return data


def avg_disbursement_missing_value(data):
    data = data.fillna({'Avg_Disbursement_Amount_Bureau': 0})
    data = data.rename(
        columns={
            'Avg_Disbursement_Amount_Bureau': 'avg_disbursement_amount_bureau'
        }
    )
    return data


# Handling Float DataTypes
def process_float_datatypes(data):

    # Cropping Density
    data = data.rename(
        columns={
            'Kharif Seasons  Cropping density in 2022': 'kharif_seasons_cropping_density_2022',
            'Rabi Seasons Cropping density in 2022': 'rabi_seasons_cropping_density_2022',
            
        }
    )

    # Agricultural Performance
    data['kharif_season_agricultural_performance_2022_2021_growth'] = data.apply(
        lambda x: (x['Kharif Seasons  Agricultural performance in 2022'] - x['Kharif Seasons Agricultural performance in 2021']) / (x['Kharif Seasons  Agricultural performance in 2022'] + 1e-5),
        axis=1
    )

    data['rabi_season_agricultural_performance_2022_2021_growth'] = data.apply(
        lambda x: (x['Rabi Seasons Agricultural performance in 2022'] - x['Rabi Seasons Agricultural performance in 2021']) / (x['Rabi Seasons Agricultural performance in 2022'] + 1e-5),
        axis=1
    )

    data['kharif_season_agricultural_performance_2021_2020_growth'] = data.apply(
        lambda x: (x['Kharif Seasons Agricultural performance in 2021'] - x['Kharif Seasons Agricultural performance in 2020']) / (x['Kharif Seasons Agricultural performance in 2021'] + 1e-5),
        axis=1
    )

    data['rabi_season_agricultural_performance_2021_2020_growth'] = data.apply(
        lambda x: (x['Rabi Seasons Agricultural performance in 2021'] - x['Rabi Seasons Agricultural performance in 2020']) / (x['Rabi Seasons Agricultural performance in 2021'] + 1e-5),
        axis=1
    )

    data = data.rename(
        columns={
            'Kharif Seasons  Agricultural performance in 2022': 'kharif_seasons_agricultural_performance_2022',
            'Rabi Seasons Agricultural performance in 2022': 'rabi_seasons_agricultural_performance_2022'
        }
    )


    data['kharif_recency_agricultural_performeance'] = data[[
        "Kharif Seasons Agricultural performance in 2021",
        "Kharif Seasons Agricultural performance in 2020"
    ]].mean(axis=1)

    data['rabi_recency_agricultural_performance'] = data[[
        "Rabi Seasons Agricultural performance in 2021",
        "Rabi Seasons Agricultural performance in 2020"
    ]].mean(axis=1)


    # Agricultural Score
    data['kharif_season_agriculture_score_2022_2021_growth'] = data.apply(
        lambda x: (x['Kharif Seasons  Agricultural Score in 2022'] - x['Kharif Seasons Agricultural Score in 2021']) - (x['Kharif Seasons  Agricultural Score in 2022'] + 1e-5),
        axis=1
    )

    data['rabi_season_agriculture_score_2022_2021_growth'] = data.apply(
        lambda x: (x['Rabi Seasons Agricultural Score in 2022'] - x['Rabi Seasons Agricultural Score in 2021']) - (x['Rabi Seasons Agricultural Score in 2022'] + 1e-5),
        axis=1
    )

    data['kharif_season_agriculture_score_2021_2020_growth'] = data.apply(
        lambda x: (x['Kharif Seasons Agricultural Score in 2021'] - x['Kharif Seasons Agricultural Score in 2020']) - (x['Kharif Seasons Agricultural Score in 2021'] + 1e-5),
        axis=1
    )

    data['rabi_season_agriculture_score_2021_2020_growth'] = data.apply(
        lambda x: (x['Rabi Seasons Agricultural Score in 2021'] - x['Rabi Seasons Agricultural Score in 2020']) - (x['Rabi Seasons Agricultural Score in 2021'] + 1e-5),
        axis=1
    )

    data = data.rename(
        columns={
            'Kharif Seasons  Agricultural Score in 2022': 'kharif_seasons_agricultural_score_2022',
            'Rabi Seasons Agricultural Score in 2022': 'rabi_seasons_agricultiral_score_2022'
        }
    )

    data['kharif_recency_agricultural_score'] = data[[
        "Kharif Seasons Agricultural Score in 2021",
        "Kharif Seasons Agricultural Score in 2020"
    ]].mean(axis=1)

    data['rabi_recency_agricultural_score'] = data[[
        "Rabi Seasons Agricultural Score in 2021",
        "Rabi Seasons Agricultural Score in 2020"
    ]].mean(axis=1)

    # Irrigated Area
    data['kharif_season_irrigated_area_2022_2021_growth'] = data.apply(
        lambda x: (x['Kharif Seasons  Irrigated area in 2022'] - x['Kharif Seasons Kharif Season Irrigated area in 2021']) - (x['Kharif Seasons  Irrigated area in 2022'] + 1e-5),
        axis=1
    )

    data['rabi_season_irrigated_area_2022_2021_growth'] = data.apply(
        lambda x: (x['Rabi Seasons  Season Irrigated area in 2022'] - x['Rabi Seasons Kharif Season Irrigated area in 2021']) - (x['Rabi Seasons  Season Irrigated area in 2022'] + 1e-5),
        axis=1
    )

    data['kharif_season_irrigated_area_2021_2020_growth'] = data.apply(
        lambda x: (x['Kharif Seasons Kharif Season Irrigated area in 2021'] - x['Kharif Seasons Kharif Season Irrigated area in 2020']) - (x['Kharif Seasons Kharif Season Irrigated area in 2021'] + 1e-5),
        axis=1
    )

    data['rabi_season_irrigated_area_2021_2020_growth'] = data.apply(
        lambda x: (x['Rabi Seasons Kharif Season Irrigated area in 2021'] - x['Rabi Seasons Kharif Season Irrigated area in 2020']) - (x['Rabi Seasons Kharif Season Irrigated area in 2021'] + 1e-5),
        axis=1
    )
    data = data.rename(
        columns={
            'Kharif Seasons  Irrigated area in 2022': 'kharif_seasons_irrigated_area_2022',
            'Rabi Seasons  Season Irrigated area in 2022': 'rabi_seasons_irrigated_area_2022'
        }
    )

    # Ground Water Thickness
    data['kharif_groundwater_thickness_2022_2021_growth'] = data.apply(
        lambda x: (x['Kharif Seasons  Seasonal average groundwater thickness (cm) in 2022'] - x['Kharif Seasons Seasonal average groundwater thickness (cm) in 2021']) / (x['Kharif Seasons  Seasonal average groundwater thickness (cm) in 2022'] + 1e-5),
        axis=1
    )

    data['rabi_groundwater_thickness_2022_2021_growth'] = data.apply(
        lambda x: (x['Rabi Seasons Seasonal average groundwater thickness (cm) in 2022'] - x['Rabi Seasons Seasonal average groundwater thickness (cm) in 2021']) / (x['Rabi Seasons Seasonal average groundwater thickness (cm) in 2022'] + 1e-5),
        axis=1
    )

    data['kharif_groundwater_thickness_2021_2020_growth'] = data.apply(
        lambda x: (x['Kharif Seasons Seasonal average groundwater thickness (cm) in 2021'] - x['Kharif Seasons Seasonal average groundwater thickness (cm) in 2020']) / (x['Kharif Seasons Seasonal average groundwater thickness (cm) in 2021'] + 1e-5),
        axis=1
    )

    data['rabi_groundwater_thickness_2021_2020_growth'] = data.apply(
        lambda x: (x['Rabi Seasons Seasonal average groundwater thickness (cm) in 2021'] - x['Rabi Seasons Seasonal average groundwater thickness (cm) in 2020']) / (x['Rabi Seasons Seasonal average groundwater thickness (cm) in 2021'] + 1e-5),
        axis=1
    )

    data = data.rename(
        columns={
            'Kharif Seasons  Seasonal average groundwater thickness (cm) in 2022': 'kharif_season_avg_groundwater_thickness_cm_2022',
            'Rabi Seasons Seasonal average groundwater thickness (cm) in 2022': 'rabi_season_avg_groundwater_thickness_cm_2022'
        }
    )

    data['kharif_season_recency_avg_groundwater_thickness_cm'] = data[[
        "Kharif Seasons Seasonal average groundwater thickness (cm) in 2021",
        "Kharif Seasons Seasonal average groundwater thickness (cm) in 2020"
    ]].mean(axis=1)

    data['rabi_season_recency_avg_groundwater_thickness_cm'] = data[[
        "Rabi Seasons Seasonal average groundwater thickness (cm) in 2021",
        "Rabi Seasons Seasonal average groundwater thickness (cm) in 2020"
    ]].mean(axis=1)

    # Ground Water Replenishment
    data['kharif_season_groundwater_replenishment_rate_2022_2021_growth'] = data.apply(
        lambda x: (x['Kharif Seasons  Seasonal average groundwater replenishment rate (cm) in 2022'] - x['Kharif Seasons Seasonal average groundwater replenishment rate (cm) in 2021']) / (x['Kharif Seasons  Seasonal average groundwater replenishment rate (cm) in 2022'] + 1e-5),
        axis=1
    )

    data['rabi_season_groundwater_replenishment_rate_2022_2021_growth'] = data.apply(
        lambda x: (x['Rabi Seasons Seasonal average groundwater replenishment rate (cm) in 2022'] - x['Rabi Seasons Seasonal average groundwater replenishment rate (cm) in 2021']) / (x['Rabi Seasons Seasonal average groundwater replenishment rate (cm) in 2022'] + 1e-5),
        axis=1
    )

    data['kharif_season_groundwater_replenishment_rate_2021_2020_growth'] = data.apply(
        lambda x: (x['Kharif Seasons Seasonal average groundwater replenishment rate (cm) in 2021'] - x['Kharif Seasons Seasonal average groundwater replenishment rate (cm) in 2020']) / (x['Kharif Seasons Seasonal average groundwater replenishment rate (cm) in 2021'] + 1e-5),
        axis=1
    )

    data['rabi_season_groundwater_replenishment_rate_2021_2020_growth'] = data.apply(
        lambda x: (x['Rabi Seasons Seasonal average groundwater replenishment rate (cm) in 2021'] - x['Rabi Seasons Seasonal average groundwater replenishment rate (cm) in 2020']) / (x['Rabi Seasons Seasonal average groundwater replenishment rate (cm) in 2021'] + 1e-5),
        axis=1
    )
    data = data.rename(
        columns={
            "Kharif Seasons  Seasonal average groundwater replenishment rate (cm) in 2022": 'kharif_season_avg_groundwater_replenishment_rate_cm_2022',
            "Rabi Seasons Seasonal average groundwater replenishment rate (cm) in 2022": 'rabi_season_avg_groundwater_replenishment_rate_cm_2022'
        }
    )

    data['kharif_season_recency_groundwater_replenishment_rate_cm'] = data[[
        "Kharif Seasons Seasonal average groundwater replenishment rate (cm) in 2021",
        "Kharif Seasons Seasonal average groundwater replenishment rate (cm) in 2020"
    ]].mean(axis=1)

    data['rabi_season_recency_groundwater_replenishment_rate_cm'] = data[[
        "Rabi Seasons Seasonal average groundwater replenishment rate (cm) in 2021",
        "Rabi Seasons Seasonal average groundwater replenishment rate (cm) in 2020"
    ]].mean(axis=1)

    # Average Rainfall
    data['kharif_average_rainfall_2022_2021_growth'] = data.apply(
        lambda x: (x['K022-Seasonal Average Rainfall (mm)'] - x['K021-Seasonal Average Rainfall (mm)']) / (x['K022-Seasonal Average Rainfall (mm)'] + 1e-5),
        axis=1
    )

    data['rabi_average_rainfall_2022_2021_growth'] = data.apply(
        lambda x: (x['R022-Seasonal Average Rainfall (mm)'] - x['R021-Seasonal Average Rainfall (mm)']) / (x['R022-Seasonal Average Rainfall (mm)'] + 1e-5),
        axis=1
    )

    # pearl_train_df['kharif_average_rainfall_2021_2020_growth'] = pearl_train_df.apply(
    #     lambda x: (x['K021-Seasonal Average Rainfall (mm)'] - x['']) / (x['K021-Seasonal Average Rainfall (mm)'] + 1e-5),
    #     axis=1
    # )

    data['rabi_average_rainfall_2021_2020_growth'] = data.apply(
        lambda x: (x['R021-Seasonal Average Rainfall (mm)'] - x['R020-Seasonal Average Rainfall (mm)']) / (x['R021-Seasonal Average Rainfall (mm)'] + 1e-5),
        axis=1
    )
    data = data.rename(
        columns={
            'K022-Seasonal Average Rainfall (mm)': 'kharif_season_avg_rainfall_mm_2022',
            'R022-Seasonal Average Rainfall (mm)': 'rabi_season_avg_rainfall_mm_2022'
        }
    )

    data['kharif_season_recency_avg_rainfall_mm'] = data[[
        "K021-Seasonal Average Rainfall (mm)"
    ]].mean(axis=1)

    data['rabi_season_recency_avg_rainfall_mm'] = data[[
        "R021-Seasonal Average Rainfall (mm)",
        "R020-Seasonal Average Rainfall (mm)"
    ]].mean(axis=1)

    # FarmerID level Features
    data = data.rename(
        columns={
            'Perc_of_house_with_6plus_room': 'perc_of_house_with_6plus_room',
            'Women_15_19_Mothers_or_Pregnant_at_time_of_survey': 'women_15_19_mothers_or_pregnant_at_time_of_survey',
            'perc_of_pop_living_in_hh_electricity': 'perc_of_pop_living_in_hh_electricity',
            'perc_Households_with_Pucca_House_That_Has_More_Than_3_Rooms': 'perc_households_with_pucca_house_that_has_more_than_3_rooms',
            'mat_roof_Metal_GI_Asbestos_sheets': 'mat_roof_metal_gi_asbestos_sheets',
            'perc_of_Wall_material_with_Burnt_brick': 'perc_of_wall_material_with_burnt_brick',
            'Households_with_improved_Sanitation_Facility': 'households_with_improved_sanitation_facility',
            'perc_Households_do_not_have_KCC_With_The_Credit_Limit_Of_50k': 'perc_households_do_not_have_kcc_with_the_credit_limit_of_50k',
            'K022-Total Geographical Area (in Hectares)-': 'total_geographical_area_in_hectare',
            'K022-Net Agri area (in Ha)-': 'net_agri_area_in_hectare',
            'K022-Net Agri area (% of total geog area)-': 'perc_net_agri_area_in_hectare',
            'Land Holding Index source (Total Agri Area/ no of people)': 'land_holding_index_source',
            'Road density (Km/ SqKm)': 'road_density'
        }
    )

    return data


def create_water_body_feature(x):
    if 'reservoir' in x['water_body']:
        x['waterbody_reservoir'] = 1

    if 'river' in x['water_body']:
        x['waterbody_river'] = 1

    if 'riverbank' in x['water_body']:
        x['waterbody_riverbank'] = 1

    if 'water' in x['water_body']:
        x['waterbody_water'] = 1

    if 'wetland' in x['water_body']:
        x['waterbody_wetland'] = 1

    return x

def process_object_datatypes(data):

    data = data.rename(columns={
        'State': 'state',
        'REGION': 'region'
    })
    
    data['soil_type'] = data['Kharif Seasons  Type of soil in 2022']

    # soil_type_mapping = {
    #     'Deep Black soils (with shallow and medium Black Soils as inclusion)': 'deep_black_soils',
    #     'Mixed Red and Black Soils': 'mixed_red_and_black_soils',
    #     'Shallow Black Soils (with medium and deep Black Soils as  inclusion)': 'shallow_black_soils',
    #     'Red and lateritic Soils': 'red_and_lateritic_soils',
    #     'Red loamy Soils': 'red_loamy_soils',
    #     'Coastal and Deltaic Alluvium derived Soils': 'coastal_and_deltaic_alluvim_derived_soils',
    #     'Alluvial-derived Soils (with saline phases)': 'alluvial_derived_soils',
    #     'Desert (saline) Soils': 'desert'
    # }

    data['soil_type'] = data['soil_type'].map(soil_type_mapping)

    data['kharif_ambient_temperature_min_2022'] = data['K022-Ambient temperature (min & max)'].map(lambda x: x.split('/')[0])
    data['kharif_ambient_temperature_max_2022'] = data['K022-Ambient temperature (min & max)'].map(lambda x: x.split('/')[1])

    data['rabi_ambient_temperature_min_2022'] = data['R022-Ambient temperature (min & max)'].map(lambda x: x.split('/')[0])
    data['rabi_ambient_temperature_max_2022'] = data['R022-Ambient temperature (min & max)'].map(lambda x: x.split('/')[1])

    data['kharif_ambient_temperature_min_2021'] = data['K021-Ambient temperature (min & max)'].map(lambda x: x.split('/')[0])
    data['kharif_ambient_temperature_max_2021'] = data['K021-Ambient temperature (min & max)'].map(lambda x: x.split('/')[1])

    data['rabi_ambient_temperature_min_2021'] = data['R021-Ambient temperature (min & max)'].map(lambda x: x.split('/')[0])
    data['rabi_ambient_temperature_max_2021'] = data['R021-Ambient temperature (min & max)'].map(lambda x: x.split('/')[1])

    data['rabi_ambient_temperature_min_2020'] = data['R020-Ambient temperature (min & max)'].map(lambda x: x.split('/')[0])
    data['rabi_ambient_temperature_max_2020'] = data['R020-Ambient temperature (min & max)'].map(lambda x: x.split('/')[1])

    # Village Category: Agricultural Parameters
    data = data.rename(
        columns={
            'K022-Village category based on Agri parameters (Good, Average, Poor)': 'kharif_vill_cat_on_agri_params_2022',
            'R022-Village category based on Agri parameters (Good, Average, Poor)': 'rabi_vill_cat_on_agri_params_2022'
        }
    )

    # Village Category: Socio-Economic Parameters
    data['village_cat_on_socio_economic_params'] = data['Village category based on socio-economic parameters (Good, Average, Poor)']

    # Water Bodies
    data['water_body'] = data['Kharif Seasons  Type of water bodies in hectares 2022'].apply(
        lambda x: eval(x)[0].split(',') if eval(x)[0] != None else [eval(x)[0]]
    )
    water_bodies = ['reservoir', 'river', 'riverbank', 'water', 'wetland']

    data['waterbody_reservoir'] = 0
    data['waterbody_river'] = 0
    data['waterbody_riverbank'] = 0
    data['waterbody_water'] = 0
    data['waterbody_wetland'] = 0

    data = data.apply(lambda x: create_water_body_feature(x), axis=1)

    # Ecological Sub-Zone
    data['ecological_subzone'] = data['Kharif Seasons  Agro Ecological Sub Zone in 2022'].copy()
    # ecological_subzone_mapping = {
    #     'CENTRAL HIGHLANDS (MALWA AND BUNDELKHAND)  HOT SUBHUMID (DRY) ECO-REGION': 'eco_subzone_1',
    #     'DECCAN PLATEAU  (TELANGANA) AND EASTERN GHATS  HOT SEMI ARID ECO-REGION': 'eco_subzone_2',
    #     'CENTRAL HIGHLANDS ( MALWA )  GUJARAT PLAIN AND KATHIAWAR PENINSULA  SEMI-ARID ECO-REGION': 'eco_subzone_3',
    #     'DECCAN PLATU  HOT SEMI-ARID ECO-REGION': 'eco_subzone_4',
    #     'KARNATAKA PLATEAU (RAYALSEEMA AS INCLUSION)': 'eco_subzone_5',
    #     'EASTERN PLATEAU (CHHOTANAGPUR) AND EASTERN GHATS  HOT SUBHUMID ECO-REGION': 'eco_subzone_6',
    #     'EASTERN GHATS AND TAMIL NADU UPLANDS AND DECCAN (K ARNATAKA) PLATEAU  HOT SEMI-ARID ECO-REGION': 'eco_subzone_7',
    #     'EASTERN COASTAL PLAIN  HOT SUBHUMID TO SEMI-ARID EGO-REGION': 'eco_subzone_8',
    #     'NORTHERN PLAIN (AND CENTRAL HIGHLANDS) INCLUDING ARAVALLIS  HOT SEMI-ARID EGO-REGION': 'eco_subzone_9',
    #     'NORTHERN PLAIN  HOT SUBHUMID (DRY) ECO-REGION': 'eco_subzone_10',
    #     'WESTERN PLAIN  KACHCHH AND PART OF KATHIAWAR PENINSULA, HOT ARID ECO-REGION': 'eco_subzone_11',
    #     'WESTERN GHATS AND COASTAL PLAIN  HOT HUMID-PERHUMID ECO-REGION': 'eco_subzone_12',
    # }

    data['ecological_subzone'] = data['ecological_subzone'].map(ecological_subzone_mapping)
    return data


def process_int_datatypes(data):
    data = data.rename(
        columns={
            'Zipcode': 'zipcode',
            'No_of_Active_Loan_In_Bureau': 'no_of_active_loan_in_bureau',
            'Non_Agriculture_Income': 'non_agriculture_income'
        }
    )

    return data


if __name__=="__main__":

    train=False
    exp_dir='exp_v1'
    if train:
        data_fname='train_v0.csv'
    else:
        data_fname='test_v0.csv'

    os.makedirs(PROCESSED_DATA_DIR / exp_dir, exist_ok=True)
    
    final_float_columns = [
        'FarmerID',
        'perc_of_house_with_6plus_room',
        'women_15_19_mothers_or_pregnant_at_time_of_survey',
        'perc_of_pop_living_in_hh_electricity',
        'perc_households_with_pucca_house_that_has_more_than_3_rooms',
        'mat_roof_metal_gi_asbestos_sheets',
        'perc_of_wall_material_with_burnt_brick',
        'households_with_improved_sanitation_facility',
        'perc_households_do_not_have_kcc_with_the_credit_limit_of_50k',
        'total_geographical_area_in_hectare',
        'net_agri_area_in_hectare',
        'perc_net_agri_area_in_hectare',
        'land_holding_index_source',
        'road_density',
        # Average Disbursement Amount Bureau
        'avg_disbursement_amount_bureau',
        # Cropping Density
        'kharif_seasons_cropping_density_2022',
        'rabi_seasons_cropping_density_2022',
        # Agricultural Performance
        'kharif_seasons_agricultural_performance_2022',
        'rabi_seasons_agricultural_performance_2022',
        # 'kharif_recency_agricultural_performeance',
        # 'rabi_recency_agricultural_performance',
        # Agricultural Score
        'kharif_seasons_agricultural_score_2022',
        'rabi_seasons_agricultiral_score_2022',
        # 'kharif_recency_agricultural_score',
        # 'rabi_recency_agricultural_score',
        # Irrigated Area
        'kharif_seasons_irrigated_area_2022',
        'rabi_seasons_irrigated_area_2022',
        # Ground Water Thickness
        'kharif_season_avg_groundwater_thickness_cm_2022',
        'rabi_season_avg_groundwater_thickness_cm_2022',
        # 'kharif_season_recency_avg_groundwater_thickness_cm',
        # 'rabi_season_recency_avg_groundwater_thickness_cm',
        # Ground Water Replenishment
        'kharif_season_avg_groundwater_replenishment_rate_cm_2022',
        'rabi_season_avg_groundwater_replenishment_rate_cm_2022',
        # 'kharif_season_recency_groundwater_replenishment_rate_cm',
        # 'rabi_season_recency_groundwater_replenishment_rate_cm',
        # Average Rainfall
        'kharif_season_avg_rainfall_mm_2022',
        'rabi_season_avg_rainfall_mm_2022',
        # 'kharif_season_recency_avg_rainfall_mm',
        # 'rabi_season_recency_avg_rainfall_mm',

        # Derived Features
        # Average Rainfall
        'kharif_average_rainfall_2022_2021_growth',
        'rabi_average_rainfall_2022_2021_growth',
        'rabi_average_rainfall_2021_2020_growth',
        # 'kharif_season_avg_rainfall_mm_2022',
        # 'rabi_season_avg_rainfall_mm_2022',
        # Average Groundwater Replenishment
        'kharif_season_groundwater_replenishment_rate_2022_2021_growth',
        'rabi_season_groundwater_replenishment_rate_2022_2021_growth',
        'kharif_season_groundwater_replenishment_rate_2021_2020_growth',
        'rabi_season_groundwater_replenishment_rate_2021_2020_growth',
        # 'kharif_season_avg_groundwater_replenishment_rate_cm_2022',
        # 'rabi_season_avg_groundwater_replenishment_rate_cm_2022',
        # Groundwater Thickness
        # 'kharif_season_avg_groundwater_thickness_cm_2022',
        # 'rabi_season_avg_groundwater_thickness_cm_2022',
        'kharif_groundwater_thickness_2022_2021_growth',
        'rabi_groundwater_thickness_2022_2021_growth',
        'kharif_groundwater_thickness_2021_2020_growth',
        'rabi_groundwater_thickness_2021_2020_growth',
        # Irrigated Area
        # 'kharif_season_irrigated_area_2022_2021_growth',
        # 'rabi_season_irrigated_area_2022_2021_growth',
        # 'kharif_season_irrigated_area_2021_2020_growth',
        # 'rabi_season_irrigated_area_2021_2020_growth',
        # 'kharif_seasons_irrigated_area_2022',
        # 'rabi_seasons_irrigated_area_2022',
        # Agricultural Performance
        'kharif_season_agricultural_performance_2022_2021_growth',
        'rabi_season_agricultural_performance_2022_2021_growth',
        'kharif_season_agricultural_performance_2021_2020_growth',
        'rabi_season_agricultural_performance_2021_2020_growth',
        # 'kharif_seasons_agricultural_performance_2022',
        # 'rabi_seasons_agricultural_performance_2022',
        # Agricultural Score
        'kharif_season_agriculture_score_2022_2021_growth',
        'rabi_season_agriculture_score_2022_2021_growth',
        'kharif_season_agriculture_score_2021_2020_growth',
        'rabi_season_agriculture_score_2021_2020_growth',
        # 'kharif_seasons_agricultural_score_2022',
        # 'rabi_seasons_agricultiral_score_2022',
        # Cropping Density
        # 'cropping_density_kharif_2022_2021_growth',
        # 'cropping_density_rabi_2022_2021_growth',
        # 'cropping_density_kharif_2021_2020_growth',
        # 'cropping_density_rabi_2021_2020_growth',
        # 'kharif_seasons_cropping_density_2022',
        # 'rabi_seasons_cropping_density_2022'
        # Ambient Temperatures
        'kharif_ambient_temperature_min_2022',
        'kharif_ambient_temperature_max_2022',
        'rabi_ambient_temperature_min_2022',
        'rabi_ambient_temperature_max_2022',
        'kharif_ambient_temperature_min_2021',
        'kharif_ambient_temperature_max_2021',
        'rabi_ambient_temperature_min_2021',
        'rabi_ambient_temperature_max_2021',
        'rabi_ambient_temperature_min_2020',
        'rabi_ambient_temperature_max_2020',
    ]

    final_object_columns = [
        'FarmerID',
        'state',
        'region',
        # Soil Types
        'soil_type',
        # 'soil_type_deep_black_soils',
        # 'soil_type_mixed_red_and_black_soils',
        # 'soil_type_shallow_black_soils',
        # 'soil_type_red_and_lateritic_soils',
        # 'soil_type_red_loamy_soils',
        # 'soil_type_coastal_and_deltaic_alluvim_derived_soils',
        # 'soil_type_alluvial_derived_soils',
        # 'soil_type_desert',
        # Kharif Season Village Category based on Agri Parameters
        'kharif_vill_cat_on_agri_params_2022',
        # 'kharif_vill_cat_agri_param_average',
        # 'kharif_vill_cat_agri_param_poor',
        # Rabi Season Village Category based on Agri Parameters
        'rabi_vill_cat_on_agri_params_2022',
        # 'rabi_vill_cat_agri_param_average',
        # 'rabi_vill_cat_agri_param_poor',
        # Village Category based on Socio-Economic Parameters
        'village_cat_on_socio_economic_params',
        # 'vill_cat_on_socio_econ_param_avg',
        # 'vill_cat_on_socio_econ_param_poor',
        # Water Body
        'waterbody_reservoir',
        'waterbody_river',
        'waterbody_riverbank',
        'waterbody_water',
        'waterbody_wetland',
        # Ecological Sub-Zone
        'ecological_subzone',
        # 'eco_subzone_1',
        # 'eco_subzone_2',
        # 'eco_subzone_3',
        # 'eco_subzone_4',
        # 'eco_subzone_5',
        # 'eco_subzone_6',
        # 'eco_subzone_7',
        # 'eco_subzone_8',
        # 'eco_subzone_9',
        # 'eco_subzone_10',
        # 'eco_subzone_11',
        # 'eco_subzone_12'
    ]

    final_int_columns = [
        'FarmerID', 
        # 'zipcode', 
        'no_of_active_loan_in_bureau', 
        'non_agriculture_income'
    ]

    target_column = ['Target_Variable/Total Income']

    print('Load Data')
    pearl_df = load_data(train=train)
    print('Handling Missing Data')
    pearl_df = avg_disbursement_missing_value(pearl_df)
    print('Handling Float Data')
    pearl_df = process_float_datatypes(pearl_df)
    print('Handling Object Data')
    pearl_df = process_object_datatypes(pearl_df)
    print('Handling Integer Data')
    pearl_df = process_int_datatypes(pearl_df)

    print('Derived Features')
    print("Active loan bins")
    pearl_df, feat_cols_active_loan_bins = create_nos_active_loan_bins(pearl_train_df=pearl_df, is_train=train)
    # active_loan_df = pearl_df[feat_cols_active_loan_bins]
    final_int_columns.extend(feat_cols_active_loan_bins[1:])

    # print('Ownership')
    # pearl_df, feat_cols_ownership = ownership_encoding(data=pearl_df)
    # # ownership_df = pearl_df[feat_cols_ownership]
    # final_int_columns.extend(feat_cols_ownership[1:])

    # print("Address type")
    # pearl_df, feat_cols_address_type = addresstype_encoding(data=pearl_df)
    # # address_type_df = pearl_df[feat_cols_address_type]
    # final_int_columns.extend(feat_cols_address_type[1:])

    print('Village Clubbing')
    pearl_df, feat_cols_village_clubbing = village_clubbing(data=pearl_df, exp_name=exp_dir, is_train=train)
    # village_clubbing_df = pearl_df[feat_cols_village_clubbing]
    final_object_columns.extend(feat_cols_village_clubbing[1:]) # Encode with Label Encoder

    print('Village Non-Agri Income Bins')
    pearl_df, feat_cols_village_non_agri_income_bins = village_non_agri_income_bins(
        data=pearl_df,
        is_train=train
    )
    # village_non_agri_income_bin_df = pearl_df[feat_cols_village_non_agri_income_bins]
    final_int_columns.extend(feat_cols_village_non_agri_income_bins[1:])

    print('Village Agri Land')
    pearl_df, feat_cols_village_agri_lands = village_agri_land_bins(data=pearl_df, is_train=train)
    # village_agri_land_df = pearl_df[feat_cols_village_agri_lands]
    final_int_columns.extend(feat_cols_village_agri_lands[1:]) 

    print('District clubbing')
    pearl_df, feat_cols_district_clubbing = district_clubbing(data=pearl_df, exp_name=exp_dir, is_train=train)
    # district_clubbing_df = pearl_df[feat_cols_district_clubbing]
    final_object_columns.extend(feat_cols_district_clubbing[1:]) # Encode with Label Encoder

    print('District Non Agri income bins')
    pearl_df, feat_cols_district_non_agri_income_bins = district_non_ari_income_bins(data=pearl_df, is_train=train)
    # district_non_agri_income_df = pearl_df[feat_cols_district_non_agri_income_bins]
    final_int_columns.extend(feat_cols_district_non_agri_income_bins[1:])

    print('District agri land bins')
    pearl_df, feat_cols_district_agri_land_bins = district_agri_land_bins(data=pearl_df, is_train=train)
    # district_agri_land_df = pearl_df[feat_cols_district_agri_land_bins]
    final_int_columns.extend(feat_cols_district_agri_land_bins[1:])


    pearl_float_df = pearl_df[final_float_columns]
    pearl_object_df = pearl_df[final_object_columns]
    pearl_int_df = pearl_df[final_int_columns]

    pearl_data_df = pearl_float_df.merge(
        right=pearl_object_df,
        on='FarmerID'
    )
    pearl_data_df = pearl_data_df.merge(
        right=pearl_int_df,
        on="FarmerID"
    )

    if train:
        target_columns = ['FarmerID', target_column[0]]
        pearl_data_df = pearl_data_df.merge(
            right=pearl_df[target_columns],
            on='FarmerID'
        )
    
    # Label Encoder
    encoder = {}
    categorical_columns = [
        'soil_type', 'state', 'region',
        'village_en_name', 'district_en_name',
        'kharif_vill_cat_on_agri_params_2022', 
        'rabi_vill_cat_on_agri_params_2022', 
        'village_cat_on_socio_economic_params', 
        'ecological_subzone'
    ]

    if train:
        encoder = {}
        for col in categorical_columns:
            le = LabelEncoder()
            pearl_data_df[col] = le.fit_transform(pearl_data_df[col])
            encoder[col] = le

        # Saving Processed Data
        processing_utils = {
            'training_columns': pearl_data_df.columns.tolist(),
            'target_column': target_column,
            'categorical_columns': categorical_columns,
            'label_encoder_mapping': encoder
        }
        pearl_data_df[categorical_columns] = pearl_data_df[categorical_columns].astype('category')
        os.makedirs(PROCESSED_DATA_DIR / exp_dir, exist_ok=True)
        with open(PROCESSED_DATA_DIR / exp_dir / 'processing_util.pkl', 'wb') as f:
            pickle.dump(processing_utils, f)
        processed_data_path = PROCESSED_DATA_DIR / exp_dir / data_fname
        pearl_data_df.to_csv(processed_data_path, index=False)
    else:
        with open(PROCESSED_DATA_DIR / exp_dir / 'processing_util.pkl', 'rb') as f:
            processing_utils = pickle.load(f)

        label_encoder_mapping = processing_utils['label_encoder_mapping']

        for col in label_encoder_mapping:
            le = label_encoder_mapping[col]
            pearl_data_df[col] = le.transform(pearl_data_df[col])

        processed_data_path = PROCESSED_DATA_DIR / exp_dir / data_fname
        pearl_data_df.to_csv(processed_data_path, index=False)