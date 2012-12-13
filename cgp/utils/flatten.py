# pylint: disable=C0301
"""
Flatten a nested list structure.

Source:
http://wiki.python.org/moin/ProblemSets/99%20Prolog%20Problems%20Solutions#Problem7.3AFlattenanestedliststructure
"""

def flatten(nestedList):
    """Flatten a nested list structure."""
    
    def aux(listOrItem):
        """Generator to recursively yield items."""
        if isinstance(listOrItem, list):
            for elem in listOrItem:
                for item in aux(elem):
                    yield item
        else:
            yield listOrItem
    
    return list(aux(nestedList))
