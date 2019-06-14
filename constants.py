from util import merge_dicts

STOCK_DATA_PATH = '/Users/rjh2nd/Dropbox (Personal)/StockAnalyzer/Stock Data//'
NN_TRAINING_DATA_PATH = '/Users/rjh2nd/PycharmProjects/StockAnalyzer/NN Training Data//'
FMT = "%Y-%m-%d"

#--industry specific key words--

ENERGY = ['drilling', 'oil', 'natural gas', 'petroleum', 'offshore drilling', 'solar', 'clean energy']

HEALTH_CARE = ['healthcare', 'therapeutics', 'biopharmaceutical', 'pharmaceutical', 'rare diseases', 'hospitals', 'healthcare suppliers', 'biotech', 'AI and medicine']

MINING = ['minerals', 'precious metals', 'gold', 'mining', 'silver']

AUTOMOTIVE = ['automotive', 'auto dealerships', 'cars', 'Tesla', 'used cars', 'new cars']

SEMICONDUCTOR = ['semiconductor', 'computer', 'Apple', 'Intel', 'microprocessor', 'NVIDIA']

FREIGHT = ['transportation', 'logistics', 'shipping', 'railroad', 'freight', 'long term storage', 'warehousing']

TELECOMMUNICATIONS = ['telecommunications', 'internet', 'cyber security', 'telephone', 'telephone service']

#--ticker data--

PENNY_STOCKS = { # Share price below $5.00
    'NAO': {'key_terms':  # noticesed trends with news polarity
                ['Marshall Islands', 'supply vessels', 'crew boats', 'anchor handling vessels'],
            'name': 'Nordic American Offshore'},
    'ROSE': {'key_terms':  # Notice strong trends with drilling googletrends and weak trends with news polarity
                 ENERGY,
             'name': 'Rosehill Resources'},
    'RHE': {'key_terms':
                ['AdCare Health Systems', 'healthcare', 'senior living', 'healthcare real estate', 'real estate', 'dialysis', 'Northwest Property Holdings', 'CP Nursing', 'ADK Georgia', 'Attalla Nursing'],
            'name': 'Regional Health'},
    'ASPN': {'key_terms':
                 ['aerogel', 'insulation', 'energy', 'pyrogel', 'cryogel'],
             'name': 'Aspen Aerogels'},
    'FET': {'key_terms':
                ENERGY,
            'name': 'Forum Energy Technologies'},
    'OMI': {'key_terms':
                HEALTH_CARE + ['cancer', 'immune disease', 'inflammation'],
            'name': 'Owens & Minor'},
    'IDN': {'key_terms':
                ['identity theft', 'identity fraud', 'credit card fraud', 'credit card', 'drivers license'],
            'name': 'Intellicheck'},
    'PIRS': {'key_terms':
                 ['biopharmaceutical', 'pharmaceutical', 'cancer', 'immune disease', 'oncology', 'anticalin', 'rare diseases', 'gene therapy', 'biotech'],
             'name': 'Pieris Pharmaceuticals'},
    'AGI': {'key_terms':
                MINING,
            'name': 'Alamos Gold'},
    'WPRT': {'key_terms':
                 ENERGY + FREIGHT,
             'name': 'Westport Fuel Systems'},
    'SESN': {'key_terms':
                 HEALTH_CARE + ['cancer'],
             'name': 'Sesen Bio'},
    'RAVE': {'key_terms':
                 ['pizza', 'restaurant franchise', 'food service', 'restaurant tipping'],
             'name': 'Rave Restaurant Group'},
    'CGEN': {'key_terms':
                 HEALTH_CARE + ['immune disease', 'oncology'],
             'name': 'Compugen'},
    'APPS':  {'key_terms':
                 ['mobile app', 'digital media', 'digital advertising', 'mobile user experience', 'sponsored app', 'mobile devices'],
             'name': 'Digital Turbine'},
    'OCUL':  {'key_terms':
                 HEALTH_CARE + ['cataracts'],
             'name': 'Ocular Therapeutix'},
    'LWAY':  {'key_terms':
                 ['kefir', 'plantiful', 'probugs', 'cups', 'skyr', 'cheese', 'probiotic'],
             'name': 'Lifeway Foods'},
    'PYDS':  {'key_terms':
                 ['gift cards', 'rebate cards', 'online payment', 'credit card'],
             'name': 'Payment Data Systems'},
    'IMGN':  {'key_terms':
                 HEALTH_CARE + ['oncology', 'cancer'],
             'name': 'ImmunoGen'},
    'UMC':  {'key_terms':
                 SEMICONDUCTOR,
             'name': 'United Microelectronics'},
    'AVXL':  {'key_terms':
                 HEALTH_CARE,
             'name': 'Anavex'},
    'VNTR':  {'key_terms':
                 ['water treatment', 'titanium dioxide', 'plastics', 'paper', 'printing inks', 'wood treatments'],
             'name': 'Venator Materials'},
    'TMQ':  {'key_terms':
                 MINING + ['Arctic', 'Bornite', 'trilogy', 'TMZ'],
             'name': 'Trilogy Metals'},
    'DRD':  {'key_terms':
                 MINING,
             'name': 'DRDGOLD'}
}

SMALL_CAP = { # Below $1B mkt cap
    'ARA':{'key_terms':
               HEALTH_CARE + ['dialysis', 'renal', 'nephrologist', 'kidney disease', 'kidney failure'],
           'name': 'American Renal Associates Holdings'},
    'IMTE': {'key_terms':
                 TELECOMMUNICATIONS,
             'name': 'Integrated Media Technology'},
    'VSLR': {'key_terms':
                 ENERGY + ['Tesla'],
             'name': 'Vivint Solar'},
    'OOMA': {'key_terms':
                 TELECOMMUNICATIONS,
             'name': 'Ooma'},
    'PRPO': {'key_terms':
                 HEALTH_CARE + ['AI and medicine', 'deep learning', 'meadical misdiagnosis', 'cancer'],
             'name': 'Precipio'},
    'AKTS': {'key_terms':
                 ['RF filter', 'acoustics', 'spkear', 'head phones', 'smart phone'],
             'name': 'Akoustis'},
    'BPTH': {'key_terms':
                 ['biopharmaceutical', 'pharmaceutical', 'cancer', 'immune disease', 'oncology', 'leukemia',
                  'rare diseases', 'gene therapy', 'biotech'],
             'name': 'Bio-Path Holdings'},
    'MAXR': {'key_terms':
                 ['satellites', 'robotics', 'Earth imagery', 'geospatial analytics', 'space systems'],
             'name': 'Maxar Technologies'},
    'ZYXI': {'key_terms':
                 HEALTH_CARE + ['medical devices', 'electrotherapy', 'chronic pain', 'surgery', 'rehabilitation'],
             'name': 'Zynex'},
    'RLGT': {'key_terms':
                 FREIGHT,
             'name': 'Radiant Logistics'},
    'IOTS': {'key_terms':
                 SEMICONDUCTOR,
             'name': 'Adesto Technologies'},
}

MID_CAP = { # Between $1B and $10B mkt cap
    'MAN': {'key_terms':
                ['staffing company', 'contractor', 'proffesional services', 'business services', 'administrative services'],
            'name': 'ManpowerGroup'},
    'ADNT':{'key_terms':
                AUTOMOTIVE + ['automotive seating', 'automotive supplier'],
            'name': 'Adient PLC'},
    'MRNA': {'key_terms':
                 ['biopharmaceutical', 'pharmaceutical', 'cancer', 'immune disease', 'inflammation', 'mRNA',
                  'gene therapy', 'regenerative medicine', 'oncology', 'rare diseases'],
             'name': 'Moderna'},
    'ABG': {'key_terms':
                AUTOMOTIVE,
            'name': 'Asbury Automotive Group'},
    'LAD': {'key_terms':
                AUTOMOTIVE,
            'name': 'Lithia Motors'},
    'PVG': {'key_terms':
                MINING,
            'name': 'Pretium Resources'},
    'MAG': {'key_terms':
                MINING + ['Mexican Silver Belt'],
            'name': 'MAG Silver'},
    'EIDX': {'key_terms':
                 HEALTH_CARE + ['transthyretin', 'amyloidosis', 'rare diseases', 'heart disease'] ,
             'name': 'Zynex'},
    'SHEN': {'key_terms':
                 TELECOMMUNICATIONS,
             'name': 'Shenandoah Telecommunications Company'},
    'SMI': {'key_terms':
                SEMICONDUCTOR,
            'name': 'Semiconductor Manufacturing'},
    'FATE': {'key_terms':
                 HEALTH_CARE + ['cancer', 'immune disease', 'genetic disorders'],
             'name': 'Fate Therapeutics'}
}

LARGE_CAP = { # Mkt cap above $10B
    'AMD':{'key_terms':
               SEMICONDUCTOR,
           'name': 'Advanced Micro Devices'},
    'TLSA': {'key_terms':
                 ['pharmaceutical', 'cancer', 'immune disease', 'inflammation', 'therapeutics'],
             'name': 'Tiziana Life Sciences'},
    'ENVA': {'key_terms':
                 ['financial services', 'big data', 'loans', 'financing'],
             'name': 'Enova'},
    'EXR': {'key_terms':
                ['self-storage', 'Marie Kondo', 'relocating', 'moving', 'long term storage', 'warehousing'],
            'name': 'Extra Space Storage'}
}

SMALL_TIKCERS = merge_dicts((PENNY_STOCKS, SMALL_CAP))

ALL_TICKERS = merge_dicts((PENNY_STOCKS, SMALL_CAP, MID_CAP, LARGE_CAP))