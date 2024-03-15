import pandas as pd
from data_process import write_report_csv


split = 'train'
txt_folder = 'data/mimic-cxr-data/reports/files/'
for prefix in range(10, 20):
    cxr_paths = pd.read_csv(f'data/mimic-cxr-data/files/{split}-p{prefix}.csv')['Path'].tolist()
    out_filepath = f'data/mimic-cxr-data/reports/{split}/p{prefix}.csv'
    write_report_csv(cxr_paths, txt_folder, out_filepath)
