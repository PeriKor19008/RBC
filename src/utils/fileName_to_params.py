def file_name_to_params(file_name):
    d_str = file_name[0:5]
    t_str = file_name[5:9]
    n_decimal_str = file_name[9:12]

    # Convert strings to numbers
    d = int(d_str)
    t = int(t_str)
    n_decimal = int(n_decimal_str)
    return d, t, n_decimal



