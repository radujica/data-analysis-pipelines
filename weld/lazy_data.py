class LazyData(object):
    """ Interface for the lazy representation of a piece of data """
    def eager_read(self):
        """ Reads and returns the data

        Used by LazyResult.retrieve_data. The parsers implementing this
        method are responsible for correctly storing parameters which result
        in only reading the required data. The parameters shall be updated
        through lazy_skip_columns and lazy_slice_rows.

        """
        raise NotImplementedError

    def eager_head(self, n=10):
        """ Reads the first n rows

        Used by LazyResult.head. This operation should NOT have side-effects,
        such as permanently updating the parameters used by eager_read.

        Parameters
        ----------
        n : int, optional
            how many rows to read

        """
        raise NotImplementedError

    def lazy_skip_columns(self, columns):
        """ Lazily record that these columns shall be skipped

        Used by LazyResult.update_columns

        Parameters
        ----------
        columns : list
            list of columns that shall be skipped

        """
        raise NotImplementedError

    def lazy_slice_rows(self, slice_):
        """ Lazily record that only this slice of that shall be read

        Used by LazyResult.update_rows

        Parameters
        ----------
        slice_ : slice
            which slice of data to be read when data is evaluated

        """
        raise NotImplementedError
