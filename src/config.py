from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

soil_type_mapping = {
    'Deep Black soils (with shallow and medium Black Soils as inclusion)': 'deep_black_soils',
    'Mixed Red and Black Soils': 'mixed_red_and_black_soils',
    'Shallow Black Soils (with medium and deep Black Soils as  inclusion)': 'shallow_black_soils',
    'Red and lateritic Soils': 'red_and_lateritic_soils',
    'Red loamy Soils': 'red_loamy_soils',
    'Coastal and Deltaic Alluvium derived Soils': 'coastal_and_deltaic_alluvim_derived_soils',
    'Alluvial-derived Soils (with saline phases)': 'alluvial_derived_soils',
    'Desert (saline) Soils': 'desert'
}

ecological_subzone_mapping = {
    'CENTRAL HIGHLANDS (MALWA AND BUNDELKHAND)  HOT SUBHUMID (DRY) ECO-REGION': 'eco_subzone_1',
    'DECCAN PLATEAU  (TELANGANA) AND EASTERN GHATS  HOT SEMI ARID ECO-REGION': 'eco_subzone_2',
    'CENTRAL HIGHLANDS ( MALWA )  GUJARAT PLAIN AND KATHIAWAR PENINSULA  SEMI-ARID ECO-REGION': 'eco_subzone_3',
    'DECCAN PLATU  HOT SEMI-ARID ECO-REGION': 'eco_subzone_4',
    'KARNATAKA PLATEAU (RAYALSEEMA AS INCLUSION)': 'eco_subzone_5',
    'EASTERN PLATEAU (CHHOTANAGPUR) AND EASTERN GHATS  HOT SUBHUMID ECO-REGION': 'eco_subzone_6',
    'EASTERN GHATS AND TAMIL NADU UPLANDS AND DECCAN (K ARNATAKA) PLATEAU  HOT SEMI-ARID ECO-REGION': 'eco_subzone_7',
    'EASTERN COASTAL PLAIN  HOT SUBHUMID TO SEMI-ARID EGO-REGION': 'eco_subzone_8',
    'NORTHERN PLAIN (AND CENTRAL HIGHLANDS) INCLUDING ARAVALLIS  HOT SEMI-ARID EGO-REGION': 'eco_subzone_9',
    'NORTHERN PLAIN  HOT SUBHUMID (DRY) ECO-REGION': 'eco_subzone_10',
    'WESTERN PLAIN  KACHCHH AND PART OF KATHIAWAR PENINSULA, HOT ARID ECO-REGION': 'eco_subzone_11',
    'WESTERN GHATS AND COASTAL PLAIN  HOT HUMID-PERHUMID ECO-REGION': 'eco_subzone_12',
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
