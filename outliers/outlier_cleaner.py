#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here

    for i in range(len(ages)):
        # Calculate error.
        e = (net_worths[i] - predictions[i]) ** 2

        # Add the values to cleaned data list.
        cleaned_data.append((ages[i], net_worths[i], e))

    # Sort the data by error.
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])
    
    # Remove the 10% points that have largest errors.
    cleaned_data = cleaned_data[:int(0.9 * len(cleaned_data))]

    return cleaned_data

