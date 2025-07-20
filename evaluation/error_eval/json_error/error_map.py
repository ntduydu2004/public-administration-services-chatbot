import pandas as pd

error_contain = pd.read_excel("llama_testset_response.xlsx")
main_file = pd.read_excel("llama_evaluation_results.xlsx")

error = []

for index, row in error_contain.iterrows():  
    if row['response'] == 'ERROR':
        error.append("json")
    else:
        error.append("")
        
main_file['error'] = error
main_file.to_excel("llama_evaluation_results_error.xlsx", index=False)