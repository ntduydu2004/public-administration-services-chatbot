import pandas as pd

file_name = ['gpt_evaluation_results.xlsx', 'llama_evaluation_results.xlsx', 'qwen_evaluation_results.xlsx']

results_list = []

for file in file_name:
    df = pd.read_excel(file)
       
    faithfulness = 0
    answer_relevancy = 0
    context_precision = 0
    context_recall = 0
    
    for index, row in df.iterrows():
        error_val = str(row['error']) if not pd.isna(row['error']) else ''
        if any(x in error_val for x in ['rfs', 'eng', 'json']):
            continue
        else:
            faithfulness += row['faithfulness']
            answer_relevancy += row['answer_relevancy']
            context_precision += row['context_precision']
            context_recall += row['context_recall']
            
    faithfulness = faithfulness / len(df)
    answer_relevancy = answer_relevancy / len(df)
    context_precision = context_precision / len(df)
    context_recall = context_recall / len(df)

    filename = file.replace('_evaluation_results.xlsx', '')
    
    results_list.append({
            'file_name': filename,
            'faithfulness': faithfulness,
            'answer_relevancy': answer_relevancy,
            'context_precision': context_precision,
            'context_recall': context_recall
        })
    
    print(f"File: {file}, total rows: {len(df)}")
    print(f"Faithfulness: {faithfulness}")
    print(f"Answer Relevancy: {answer_relevancy}")

res_df = pd.DataFrame(results_list)
    
res_df.to_csv('result.csv', index=False, encoding='utf-8-sig')