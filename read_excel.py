import pandas as pd

xl = pd.ExcelFile('3. Heat_Sink_Design_Ref.xlsx')
print('Sheets:', xl.sheet_names)

for sheet_name in xl.sheet_names:
    print(f'\n=== Sheet: {sheet_name} ===')
    df = pd.read_excel('3. Heat_Sink_Design_Ref.xlsx', sheet_name=sheet_name)
    print(df.to_string())
    print(f'\nShape: {df.shape}')

