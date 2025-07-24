import pandas as pd

file_name = ['gpt_accuracy_results.xlsx', 'llama_accuracy_results.xlsx', 'qwen_accuracy_results.xlsx']

results_list = []

for file in file_name:
    df = pd.read_excel(file)
       
    answer_accuracy = 0
    
    for index, row in df.iterrows():
        error_val = str(row['error']) if not pd.isna(row['error']) else ''
        if any(x in error_val for x in ['rfs', 'eng', 'json']):
            continue
        else:
            answer_accuracy += row['nv_accuracy']
            
    answer_accuracy = answer_accuracy / len(df)

    filename = file.replace('_context_results.xlsx', '')
    
    results_list.append({
            'file_name': filename,
            'accuracy': answer_accuracy
        })
    
    print(f"File: {file}, total rows: {len(df)}")
    print(f"File: {file}, answer accuracy: {answer_accuracy}")

res_df = pd.DataFrame(results_list)
    
res_df.to_csv('accuracy.csv', index=False, encoding='utf-8-sig')