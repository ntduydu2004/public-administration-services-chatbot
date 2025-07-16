import pandas as pd

def get_average_result(file_path) -> pd.DataFrame:
    # Create a list to hold the row data
    results_list = []
    for value in file_path:
        f = pd.read_excel(value)
        faitfulness = f['faithfulness'].mean()
        answer_relevancy = f['answer_relevancy'].mean()
        # Append a dictionary to the list
        results_list.append({
            'file_name': value,
            'faithfulness': faitfulness,
            'answer_relevancy': answer_relevancy
        })
    # Create the DataFrame from the list of dictionaries at the end
    result = pd.DataFrame(results_list)
    return result

# The rest of your code remains the same
file_path = ['gpt_evaluation_results.xlsx', 'qwen_evaluation_results.xlsx', 'llama_evaluation_results.xlsx']
average_result = get_average_result(file_path)
average_result.to_csv("average_evaluation_results.csv", index=False) # encoding='utf-8-sig' is often not needed here
print(average_result)