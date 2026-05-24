"""
Configuration file for Coffee Project

Contains all project paths and settings.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CLEANED_DATA_DIR = PROCESSED_DATA_DIR / "cleaned"
DICTIONARY_DIR = DATA_DIR / "dictionary"

# Raw data files
RAW_FILES = {
    'sa_var': RAW_DATA_DIR / 'SA#var.csv',
    'needstate': RAW_DATA_DIR / 'NeedstateDaydaypart.csv',
    'brandhealth': RAW_DATA_DIR / 'Brandhealth.csv',
    'brand_image': RAW_DATA_DIR / 'Brand_Image.csv',
    'segmentation': RAW_DATA_DIR / '2017Segmenttation3685case.csv'
}

# Processed data files
MERGED_DATA_FILE = PROCESSED_DATA_DIR / "merged_full.csv"

# Cleaned data files
CLEANED_FILES = {
    'customer_seg': CLEANED_DATA_DIR / 'customer_segmentation_clean.csv',
    'brand_image': CLEANED_DATA_DIR / 'brand_image_clean.csv',
    'brandhealth': CLEANED_DATA_DIR / 'brandhealth_clean.csv',
    'sa_var': CLEANED_DATA_DIR / 'sa_var_clean.csv',
    'needstate': CLEANED_DATA_DIR / 'needstate_clean.csv'
}

# Models directory
MODELS_DIR = PROJECT_ROOT / "results"

# Reports directory
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SLIDES_DIR = REPORTS_DIR / "slides"

# Model settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Logging configuration
LOG_LEVEL = "INFO"

# ============================================================================
# ENCODING CONFIGURATION
# ============================================================================

# Các cột cần DROP (không dùng cho model)
DROP_COLS = [
    'ID', 'Year',
    'TOM', 'BUMO', 'BUMO_Previous', 'MostFavourite',
    'Brand',
    'Spontaneous', 'Trial', 'P3M', 'P1M', 'Weekly', 'Daily',
    'Needstates',
    'Awareness',
    'Attribute', 'BrandImage',
    'Occupation'  # Trùng với Occupation_Group
]

# Các cột NUMERIC cần LOG transform (Visit, Spending, PPA)
LOG_NUMERIC_COLS = [
    'Visit',
    'Spending', 
    'PPA'
]

# Các cột NUMERIC thông thường (không log, chỉ scale)
NUMERIC_COLS = [
    'Age',
    'Group_size',
    'NPS_P3M',
    'Brand_Loyalty',
    'Brand_Switcher',
    'Funnel_Depth',
    'Awareness_Usage_Gap',
    'Need_is_Drinking',
    'Awareness_flag',
    'Spontaneous_flag',
    'Trial_flag',
    'P3M_flag',
    'P1M_flag',
    'Weekly_flag',
    'Daily_flag'
]

# Các cột ORDINAL (có thứ tự tự nhiên)
ORDINAL_COLS = [
    'Segmentation',
    'Age_Group_2',
    'Comprehension'
]

# Các cột CATEGORICAL (không có thứ tự)
CATEGORICAL_COLS = [
    'City',
    'Gender',
    'Occupation_Group',
    'Daypart',
    'NeedstateGroup',
    'TOM_Group',
    'BUMO_Group',
    'BUMO_Previous_Group',
    'MostFavourite_Group',
    'NPS_P3M_Group',
    'Brand_Likability'
]

# Thứ tự cho các biến ordinal
ORDINAL_CATEGORIES = [
    # Segmentation: theo mức chi tiêu từ thấp đến cao
    [
        'Seg.01 - Mass (<VND 25K)',
        'Seg.02 - Mass Asp (VND 25K - VND 59K)',
        'Seg.03 - Premium (VND 60K - VND 99K)',
        'Seg.04 - Super Premium (VND 100K+)'
    ],
    # Age_Group_2: theo độ tuổi tăng dần
    [
        '16 - 19 y.o.',
        '20 - 24 y.o.',
        '25 - 29 y.o.',
        '30 - 34 y.o.',
        '35 - 39 y.o.',
        '40 - 44 y.o.',
        '45+ y.o.'
    ],
    # Comprehension: từ không biết đến biết rõ
    [
        "Don't know",
        'Know a little',
        'Know well'
    ]
]
