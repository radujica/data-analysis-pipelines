class DataFrame(object):
    def __init__(self, data, index):
        """ Weld-ed pandas DataFrame

        Parameters
        ----------
        data : dict
            column names -> data array
        index : pdw.MultiIndex
            index

        Returns
        -------
        pdw.DataFrame

        """
        self.data = data
        self.index = index

    def __repr__(self):
        string_representation = """column names:\n\t%(columns)s\nindex names:\n\t%(indexes)s"""

        return string_representation % {'columns': self.data.keys(),
                                        'indexes': self.index.names}

    # TODO
    def evaluate(self):
        pass

    # this materializes everything right now
    def evaluate_all(self, verbose=True, decode=True, passes=None, num_threads=1,
                     apply_experimental_transforms=False):
        materialized_columns = {}
        for column in self.data.items():
            materialized_columns[column[0]] = column[1].evaluate(verbose=verbose)

        string_representation = """%(repr)s\nindex:\n\n%(index)s\ncolumns:\n\t%(columns)s"""

        return string_representation % {'repr': self.__repr__(),
                                        'index': self.index.evaluate_all(verbose, decode, passes,
                                                                         num_threads, apply_experimental_transforms),
                                        'columns': materialized_columns}
