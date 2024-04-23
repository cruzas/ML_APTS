import os
import csv
import re

# List of original CSV files to be cleaned
# original_files = [
#     './results_csv/results_APTS_W_CIFAR10_10000_2.csv',
#     './results_csv/results_APTS_W_CIFAR10_10000_4.csv',
#     './results_csv/results_APTS_W_CIFAR10_10000_6.csv'
# ]

original_files = [
    './results_csv/results_Adam_CIFAR10_10000_resnet18.csv'
]

# Function to clean cumulative times data
def clean_cum_times(cum_times_str):
    # First, ensure the square brackets exist if they were removed
    cum_times_str = cum_times_str.strip()
    if not cum_times_str.startswith('['):
        cum_times_str = '[' + cum_times_str
    if not cum_times_str.endswith(']'):
        cum_times_str = cum_times_str + ']'

    # Replace whitespaces (if any) inside brackets with commas but keep scientific notation intact
    inner_content = cum_times_str[1:-1]  # Exclude the square brackets for now
    # Properly formatted scientific notation numbers should not be altered
    # This pattern aims to preserve numbers, including those in scientific notation, while replacing other spaces
    formatted_content = re.sub(r'(?<=\d)\s+(?=[\d-])', ',', inner_content)
    # Reassemble the string with square brackets
    cleaned_str = '[' + formatted_content + ']'
    return cleaned_str

# Process each file
for file_path in original_files:
    cleaned_rows = []
    with open(file_path, mode='r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Clean the 'cum_times' column
            if 'cum_times' in row:  # Replace 'cum_times' with your actual column name
                row['cum_times'] = clean_cum_times(row['cum_times'])
            cleaned_rows.append(row)
        
    # Define new file path for cleaned data
    new_file_path = file_path.replace('.csv', '_cleaned.csv')
    
    # Write the cleaned data to a new CSV file
    with open(new_file_path, mode='w', newline='', encoding='utf-8') as new_file:
        writer = csv.DictWriter(new_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    print(f"Cleaned data written to {new_file_path}")