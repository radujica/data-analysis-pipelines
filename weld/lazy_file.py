class LazyFile(object):
    """ Interface for the lazy representation of a data file """
    def read_metadata(self):
        """ Method returning the metadata/header of the file

        This currently is not used by lazy parsing, however is a helper method
        that does need implementation to provide access only to the metadata without reading further data.

        Typically used by a parser to store this metadata and propagate it further as placeholders
        in the lazy computation graph.

        """
        raise NotImplementedError

    def read_file(self):
        """ Method reading the file

        Behaves like a handler to the file. Used by LazyResult.retrieve_file.

        """
        raise NotImplementedError
