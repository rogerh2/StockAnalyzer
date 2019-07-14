from util import merge_dicts

STOCK_DATA_PATH = '/Users/rjh2nd/Dropbox (Personal)/StockAnalyzer/Stock Data//'
DATA_PATH = '/Users/rjh2nd/Dropbox (Personal)/StockAnalyzer/data//'
NN_TRAINING_DATA_PATH = '/Users/rjh2nd/PycharmProjects/StockAnalyzer/NN Training Data//'
MODEL_PATH = '/Users/rjh2nd/PycharmProjects/StockAnalyzer/models//'
FMT = "%Y-%m-%d"

#--industry specific key words--

ENERGY = ['energy', 'drilling', 'oil', 'natural gas', 'petroleum', 'offshore drilling', 'solar', 'clean energy', 'emissions', 'nuclear power']

BIOTECH = ['healthcare', 'therapeutics', 'biopharmaceutical', 'pharmaceutical', 'rare diseases', 'hospitals', 'healthcare suppliers', 'biotech', 'AI and medicine', 'antibiotics']

CANCER = BIOTECH + ['gene therapy', 'cancer', 'oncology']

MINING = ['minerals', 'precious metals', 'gold', 'mining', 'silver', 'steel', 'lead', 'zinc']

AUTOMOTIVE = ['automotive', 'auto dealerships', 'cars', 'Tesla', 'used cars', 'new cars', 'electric vehicles', 'sustainable transportation']

SEMICONDUCTOR = ['semiconductor', 'computer', 'Apple', 'Intel', 'microprocessor', 'NVIDIA', 'integrated circuits', 'consumer electronics']

FREIGHT = ['transportation', 'logistics', 'shipping', 'railroad', 'freight', 'long term storage', 'warehousing', 'tanker']

TELECOMMUNICATIONS = ['telecommunications', 'internet', 'cyber security', 'telephone', 'telephone service', 'mobile', 'broadband networks', 'ethernet']

SENIOR_LIVING = ['senior living', 'assisted living', 'memory care', "Alzheimer's", "dementia"]

MANUFACTURING = ['steel', 'machined parts', 'OEMs', 'machine tools', 'engineering', 'industrial']

DATA_SCIENCE = ['data storage', 'data science', 'computer vision', 'smart technologies', 'machine learning', 'blockchain']

#--ticker data--

PENNY_STOCKS = { # Share price below $6.00
    'NAT': {'key_terms':  # Renamed to PSV
                ['Marshall Islands', 'supply vessels', 'crew boats', 'anchor handling vessels'],
            'name': 'Nordic American Tankers'},
    'ROSE': {'key_terms':
                 ENERGY,
             'name': 'Rosehill Resources'},
    'RHE': {'key_terms':
                SENIOR_LIVING + ['healthcare', 'healthcare real estate', 'real estate', 'dialysis'],
            'name': 'Regional Health'},
    'ASPN': {'key_terms':
                 ['aerogel', 'insulation', 'energy', 'pyrogel', 'cryogel'],
             'name': 'Aspen Aerogels'},
    'FET': {'key_terms':
                ENERGY,
            'name': 'Forum Energy Technologies'},
    'OMI': {'key_terms':
                BIOTECH + ['cancer', 'immune disease', 'inflammation'],
            'name': 'Owens & Minor'},
    'IDN': {'key_terms':
                ['identity theft', 'identity fraud', 'credit card fraud', 'credit card', 'drivers license'],
            'name': 'Intellicheck'},
    'PIRS': {'key_terms':
                 CANCER + ['immune disease', 'anticalin', 'rare diseases'],
             'name': 'Pieris Pharmaceuticals'},
    'AGI': {'key_terms':
                MINING,
            'name': 'Alamos Gold'},
    'WPRT': {'key_terms':
                 ENERGY + FREIGHT,
             'name': 'Westport Fuel Systems'},
    'SESN': {'key_terms':
                 CANCER,
             'name': 'Sesen Bio'},
    'CGEN': {'key_terms':
                 CANCER + ['immune disease'],
             'name': 'Compugen'},
    'APPS':  {'key_terms':
                 ['mobile app', 'digital media', 'digital advertising', 'mobile user experience', 'sponsored app', 'mobile devices'],
             'name': 'Digital Turbine'},
    'OCUL':  {'key_terms':
                  BIOTECH + ['cataracts'],
             'name': 'Ocular Therapeutix'},
    'LWAY':  {'key_terms':
                 ['kefir', 'plantiful', 'probugs', 'cups', 'skyr', 'cheese', 'probiotic'],
             'name': 'Lifeway Foods'},
    'PYDS':  {'key_terms':
                 ['gift cards', 'rebate cards', 'online payment', 'credit card'],
             'name': 'Payment Data Systems'},
    'IMGN':  {'key_terms':
                 CANCER,
             'name': 'ImmunoGen'},
    'UMC':  {'key_terms':
                 SEMICONDUCTOR,
             'name': 'United Microelectronics'},
    'AVXL':  {'key_terms':
                 BIOTECH,
             'name': 'Anavex'},
    'VNTR':  {'key_terms':
                 ['water treatment', 'titanium dioxide', 'plastics', 'paper', 'printing inks', 'wood treatments'],
             'name': 'Venator Materials'},
    'TMQ':  {'key_terms':
                 MINING + ['Arctic', 'Bornite', 'trilogy', 'TMZ'],
             'name': 'Trilogy Metals'},
    'ATAI':  {'key_terms':
                 ['education', 'China', 'education technologies', 'online education'],
             'name': 'ATA'},
    'VEON':  {'key_terms':
                 TELECOMMUNICATIONS,
             'name': 'Veon'},
    'RAVE': {'key_terms':
                 ['pizza', 'restaurant franchise', 'food service', 'restaurant tipping'],
             'name': 'Rave Restaurant Group'},
    'ASC': {'key_terms':
                 FREIGHT,
             'name': 'Ardmore Shipping'},
    'CSU': {'key_terms':
                 SENIOR_LIVING,
             'name': 'Capital Senior Living'},
    'BWEN': {'key_terms':
                 ENERGY + MANUFACTURING,
             'name': 'Broadwind Energy'},
    'SMIT': {'key_terms':
                 MANUFACTURING + ['automotive', 'aerospace'],
             'name': 'Schmitt Industries'},
    'FSM': {'key_terms':
                 MINING,
             'name': 'Fortuna Silver Mines'},
    'JCS': {'key_terms':
                 TELECOMMUNICATIONS,
             'name': 'Communications Systems'},
    'ALO': {'key_terms':
                MINING,
            'name': 'Alio Gold'},
    'HMY': {'key_terms':
                MINING,
            'name': 'Harmony Gold'},
    'OTLK': {'key_terms':
                 BIOTECH + ['opthalmic'],
            'name': 'Outlook Therapeutics'},
    'XELA': {'key_terms':
                DATA_SCIENCE + ['smart office'],
            'name': 'Exela Technologies'},
    'KEG': {'key_terms':
                 ENERGY + ['fishing'],
             'name': 'Key Energy Services'},
    'NBRV': {'key_terms':
                 BIOTECH,
             'name': 'Nabriva Therapeutics'},
    'EGO': {'key_terms':
                 MINING,
             'name': 'Eldorado Gold'},
    'DRYS': {'key_terms':
                 ENERGY + FREIGHT,
             'name': 'DryShips'},
    'EGY': {'key_terms':
                 ENERGY + FREIGHT,
             'name': 'Vaalco'},
    'FUV': {'key_terms':
                 ENERGY + AUTOMOTIVE,
             'name': 'Arcimoto'},
    'UEC': {'key_terms':
                 ENERGY + ['Uranium', 'Atomic', 'Iran'],
             'name': 'Uranium Energy'},
    'IAG': {'key_terms':
                 MINING,
             'name': 'Iamgold'},
    'PXLW': {'key_terms':
                 SEMICONDUCTOR + ['HD video', 'mobile displays'],
             'name': 'Pixelworks'}
}

SMALL_CAP = { # Below $1B mkt cap
    'ARA':{'key_terms':
               BIOTECH + ['dialysis', 'renal', 'nephrologist', 'kidney disease', 'kidney failure'],
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
                 BIOTECH + ['AI and medicine', 'deep learning', 'meadical misdiagnosis', 'cancer'],
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
                 BIOTECH + ['medical devices', 'electrotherapy', 'chronic pain', 'surgery', 'rehabilitation'],
             'name': 'Zynex'},
    'RLGT': {'key_terms':
                 FREIGHT,
             'name': 'Radiant Logistics'},
    'IOTS': {'key_terms':
                 SEMICONDUCTOR,
             'name': 'Adesto Technologies'},
    'CRD.A':  {'key_terms':
                 ['insurance', 'claims management solutions', 'corporate insurance plan', 'self-insured entities'],
             'name': 'Crawford & Co'}
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
                 BIOTECH + ['transthyretin', 'amyloidosis', 'rare diseases', 'heart disease'] ,
             'name': 'Zynex'},
    'SHEN': {'key_terms':
                 TELECOMMUNICATIONS,
             'name': 'Shenandoah Telecommunications Company'},
    'SMI': {'key_terms':
                SEMICONDUCTOR,
            'name': 'Semiconductor Manufacturing'},
    'FATE': {'key_terms':
                 BIOTECH + ['cancer', 'immune disease', 'genetic disorders'],
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

NEXT_TICKERS = {
    'BWEN': {'key_terms':
                 ENERGY + MANUFACTURING,
             'name': 'Broadwind Energy'}
}
# for tick in ALL_TICKERS:
#     if tick in ['ROSE', 'RHE', 'ASPN', 'FET', 'OMI', 'IDN', 'PIRS', 'AGI', 'WPRT', 'SESN', 'RAVE', 'CGEN']:
#         continue
#     else:
#         NEXT_TICKERS[tick] = ALL_TICKERS[tick]