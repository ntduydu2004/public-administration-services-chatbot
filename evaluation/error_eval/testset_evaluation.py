import pandas as pd

file_name = ['gpt_evaluation_results_error.xlsx', 'llama_evaluation_results_error.xlsx', 'qwen_evaluation_results_error.xlsx']

results_list = []

for file in file_name:
    df = pd.read_excel(file)
       
    faithfulness = 0
    answer_relevancy = 0
    
    for index, row in df.iterrows():
        error_val = str(row['error']) if not pd.isna(row['error']) else ''
        if any(x in error_val for x in ['rfs', 'eng', 'json']):
            continue
        else:
            faithfulness += row['faithfulness']
            answer_relevancy += row['answer_relevancy']
            
    faithfulness = faithfulness / len(df)
    answer_relevancy = answer_relevancy / len(df)
    filename = file.replace('_evaluation_results_error.xlsx', '')
    
    results_list.append({
            'file_name': filename,
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy
        })
    
    print(f"File: {file}, total rows: {len(df)}")
    print(f"Faithfulness: {faithfulness}")
    print(f"Answer Relevancy: {answer_relevancy}")

res_df = pd.DataFrame(results_list)
    
res_df.to_csv('result.csv', index=False, encoding='utf-8-sig')