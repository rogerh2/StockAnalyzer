STOCK_DATA_PATH = '/Users/rjh2nd/PycharmProjects/StockAnalyzer/Stock Data'
FMT = "%Y-%m-%d"
ALL_TICKERS = {
    'NAO': {'key_terms':  # noticesed trends with news polarity
                ['Marshall Islands', 'supply vessels', 'crew boats', 'anchor handling vessels'],
            'name': 'Nordic American Offshore'},
    'ROSE': {'key_terms':  # Notice strong trends with drilling googletrends and weak trends with news polarity
                 ['Delaware Basin', 'drilling', 'oil', 'natural gas', 'petroleum', 'offshore drilling'],
             'name': 'Rosehill Resources'},
    'RHE': {'key_terms':
                ['AdCare Health Systems', 'healthcare', 'senior living', 'healthcare real estate', 'real estate', 'dialysis', 'Northwest Property Holdings', 'CP Nursing', 'ADK Georgia', 'Attalla Nursing'],
            'name': 'Regional Health'},
    'MAN': {'key_terms':
                ['staffing company', 'contractor', 'proffesional services', 'business services', 'administrative services'],
            'name': 'ManpowerGroup'},
    'AMD':{'key_terms':
                ['semiconductor', 'computer', 'Apple', 'Intel', 'Microprocessor', 'NVIDIA'],
            'name': 'Advanced Micro Devices'},
    'ARA':{'key_terms':
                ['dialysis', 'renal', 'nephrologist', 'kidney disease', 'kidney failure'],
            'name': 'American Renal Associates Holdings'},
    'ADNT':{'key_terms':
                ['car', 'automotive', 'dealerships', 'used car', 'automotive seating', 'automotive supplier'],
            'name': 'Adient PLC'},
    'ASPN': {'key_terms':
                ['aerogel', 'insulation', 'energy', 'pyrogel', 'cryogel'],
             'name': 'Aspen Aerogels'},
    'TLSA': {'key_terms':
                ['Pharma', 'cancer', 'immune disease', 'inflammation', 'therapeutics'],
             'name': 'Tiziana Life Sciences'},
    'MRNA': {'key_terms':
                ['biopharmaceutical', 'pharmaceutical', 'cancer', 'immune disease', 'inflammation', 'mRNA', 'gene therapy', 'regenerative medicine', 'oncology', 'rare diseases'],
             'name': 'Moderna'},
    'IMTE': {'key_terms':
                ['telecommunications', 'cyber security', 'big data', 'IT services', 'media services'],
             'name': 'Integrated Media Technology'},
    'ENVA': {'key_terms':
                ['financial services', 'big data', 'loans', 'financing'],
         'name': 'Enova'},
    'FET': {'key_terms':
                ['drilling', 'oil', 'natural gas', 'petroleum', 'offshore drilling'],
         'name': 'Forum Energy Technologies'},
    'VSLR': {'key_terms':
                    ['Tesla', 'solar', 'clean energy', 'drilling', 'oil', 'natural gas', 'petroleum', 'offshore drilling'],
             'name': 'Vivint Solar'},
    'ABG': {'key_terms':
                 ['automotive', 'auto dealerships', 'cars', 'Tesla', 'used cars', 'new cars'],
             'name': 'Asbury Automotive Group'},
    'LAD': {'key_terms':
                 ['automotive', 'auto dealerships', 'cars', 'Tesla', 'used cars', 'new cars'],
             'name': 'Lithia Motors'},
    'OOMA': {'key_terms':
                 ['telecommunications', 'internet', 'cyber security', 'telephone', 'telephone service'],
             'name': 'Ooma'},
    'EXR': {'key_terms':
               ['self-storage', 'Marie Kondo', 'relocating', 'moving', 'long term storage', 'warehousing'],
           'name': 'Extra Space Storage'},
    'OMI': {'key_terms':
               ['Pharma', 'cancer', 'immune disease', 'inflammation', 'therapeutics', 'hospitals', 'healthcare suppliers', 'biotech'],
           'name': 'Owens & Minor'},
    'PRPO': {'key_terms':
                ['hospitals', 'AI and medicine', 'deep learning', 'meadical misdiagnosis', 'cancer', 'biotech'],
            'name': 'Precipio'},
    'AKTS': {'key_terms':
                ['RF filter', 'acoustics', 'spkear', 'head phones', 'smart phone'],
            'name': 'Akoustis'},
    'IDN': {'key_terms':
                ['identity theft', 'identity fraud', 'credit card fraud', 'credit card', 'drivers license'],
            'name': 'Intellicheck'},
    'PIRS': {'key_terms':
                ['biopharmaceutical', 'pharmaceutical', 'cancer', 'immune disease', 'oncology', 'anticalin', 'rare diseases', 'gene therapy', 'biotech'],
            'name': 'Pieris Pharmaceuticals'},
    'PVG': {'key_terms':
                ['minerals', 'precious metals', 'gold', 'mining', 'silver'],
            'name': 'Pretium Resources'},
    'AGI': {'key_terms':
                ['minerals', 'precious metals', 'gold', 'mining', 'silver'],
            'name': 'Alamos Gold'},
    'MAG': {'key_terms':
                ['minerals', 'precious metals', 'gold', 'mining', 'silver', 'Mexican Silver Belt'],
            'name': 'MAG Silver'},
    'BPTH': {'key_terms':
                ['biopharmaceutical', 'pharmaceutical', 'cancer', 'immune disease', 'oncology', 'leukemia', 'rare diseases', 'gene therapy', 'biotech'],
            'name': 'Bio-Path Holdings'},
    'MAXR': {'key_terms':
                ['satellites', 'robotics', 'Earth imagery', 'geospatial analytics', 'space systems'],
            'name': 'Maxar Technologies'},
    'ZYXI': {'key_terms':
                ['medical devices', 'electrotherapy', 'chronic pain', 'surgery', 'rehabilitation'],
            'name': 'Zynex'},
    'EIDX': {'key_terms':
                ['transthyretin', 'amyloidosis', 'rare diseases', 'heart disease', 'biotech'],
            'name': 'Zynex'},
    'RLGT': {'key_terms':
                ['transportation', 'logistics', 'shipping', 'railroad', 'freight', 'long term storage', 'warehousing'],
            'name': 'Radiant Logistics'},
    'WPRT': {'key_terms':
                ['clean energy', 'drilling', 'oil', 'natural gas', 'petroleum', 'offshore drilling', 'transportation', 'logistics'],
            'name': 'Westport Fuel Systems'},
    'SHEN': {'key_terms':
                ['telecommunications', 'internet', 'cyber security', 'telephone', 'telephone service'],
            'name': 'Shenandoah Telecommunications Company'}

}