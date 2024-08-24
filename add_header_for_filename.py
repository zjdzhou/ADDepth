def add_header_for_filenames(filenames):
    loop_nums = len(filenames)
    for index in range(loop_nums):
        filename_header = get_filename_header(filenames[index])
        filenames[index] = filename_header + "/" + filenames[index]
    return filenames


def get_filename_header(filename):
    filename_split = filename.split(" ")
    filename_header = filename_split[0][11::]
    return filename_header
