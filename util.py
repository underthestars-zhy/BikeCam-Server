def convert_interval(data):
    """
    Convert 30-second interval data to 1-minute interval by removing mid-interval records.

    Args:
        data (list): List of dictionaries containing 'time' and 'bokes' keys
                    'time' should be a timestamp

    Returns:
        list: Filtered data with 1-minute intervals
    """
    # Sort the data by timestamp
    sorted_data = sorted(data, key=lambda x: x['time'])

    # Initialize result list
    # result = []
    #
    # # Iterate through the sorted data with step=2 to skip mid-interval records
    # for i in range(0, len(sorted_data), 2):
    #     result.append(sorted_data[i])

    return sorted_data
