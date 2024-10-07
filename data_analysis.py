#!/usr/bin/env python3

import csv
import json 

def read_csv_file(file_name):
    data = []
    with open(file_name, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            data.append(row)
    return data

def read_csv_files():
    chargeback_data = read_csv_file('chargeback.csv')
    acceptance_data = read_csv_file('acceptance.csv')
    return chargeback_data, acceptance_data


def print_first_5_rows():
    chargeback_data, acceptance_data = read_csv_files()
    for row in chargeback_data[:5]:
        print(row)
    for row in acceptance_data[:5]:
        print(row)

def join_chargeback_acceptance(chargeback_data, acceptance_data):
    # Create a dictionary for quick lookup of chargeback data by external_ref
    chargeback_dict = {row['external_ref']: row for row in chargeback_data}
    # Iterate through acceptance data and add chargeback info if there's a match
    for acceptance_row in acceptance_data:
        external_ref = acceptance_row.get('external_ref')
        if external_ref in chargeback_dict:
            acceptance_row['chargeback'] = chargeback_dict[external_ref]['chargeback']
    return acceptance_data  # Return the updated acceptance data

# visualize new data in csv
def write_new_data_to_csv(full_data):
    with open('full_acceptance_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['external_ref', 'status', 'source', 'ref', 'date_time', 'state', 'cvv_provided', 'amount', 'country', 'currency', 'rates', 'chargeback'])  # Assuming these are the column names
        for row in full_data:
            writer.writerow([row['external_ref'], row.get('status', 'N/A'), row.get('source', 'N/A'), row.get('ref', 'N/A'), row.get('date_time', 'N/A'), row.get('state', 'N/A'), row.get('cvv_provided', 'N/A'), row.get('amount', 'N/A'), row.get('country', 'N/A'), row.get('currency', 'N/A'), row.get('rates', 'N/A'), row.get('chargeback', 'N/A')])


# for either chargeback/non, calculate the number of transactions and corresponding amounts
def count_states(full_data, chargebacks='FALSE'):
    declined_count = 0
    accepted_count = 0
    total_declined_amount_USD = 0
    for row in full_data:
        if row.get('chargeback') == chargebacks:
            if row.get('state') == 'DECLINED':
                declined_count += 1
                if row.get('currency') == 'USD':
                    total_declined_amount_USD += float(row.get('amount'))
                else:
                    rates = json.loads(row.get('rates'))
                    relevant_rate = rates[row.get('currency')]
                    usd_amount = float(row.get('amount'))/float(relevant_rate)
                    total_declined_amount_USD += usd_amount
            elif row.get('state') == 'ACCEPTED':
                accepted_count += 1

    return declined_count, accepted_count, total_declined_amount_USD

chargeback_data, acceptance_data = read_csv_files()
full_data = join_chargeback_acceptance(chargeback_data, acceptance_data)
# write_new_data_to_csv(full_data)
declined, accepted, total_declined_amount_USD = count_states(full_data)

# for non-chargebacked transactions....
#print(f"Declined count: {declined}, Accepted count: {accepted}, declined amount: {total_declined_amount_USD} ")


##### ------------------------- #####

# For root cause analysis 
