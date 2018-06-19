# Weld-ed netCDF4 library

Wrapper over the Python netCDF4 library which tracks computations in
Weld IR. The API is furthermore lazy by only reading data from file
when necessary.

# Example

    import netCDF4
    import netCDF4_weld

    # function to read the file/dataset; should not eagerly read everything.
    # for netCDF4, this reads the header, while the data/values are
    # read when accessed e.g. via .variables[<name>]
    def read_dataset():
        return netCDF4.Dataset(<path>)

    dsw = netCDF4_weld.Dataset(read_dataset)

    # to (lazily) describe the variable
    print(dsw.variables[<name>])
    # to (eagerly) read the data and do any recorded computations
    print(dsw.variables[<name>].evaluate())
