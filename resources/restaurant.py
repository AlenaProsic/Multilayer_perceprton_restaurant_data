# package resources

"""
    Class to represent a restaurant where the price should be predicted
"""


class Restaurant:
    def __init__(
            self,
            restaurant_id=0,
            gold="",
            predicted="",
            name="",
            cuisine="",
            location="",
            menu_text=None,
            tokens=None,
            features=None,
            menu_text_as_whole="",
    ):
        """
        Constructor
        :param restaurant_id: An int id added to every restaurant
        :param gold: A string gold annotation for a price tag of a given restaurant, e.g. $$
        :param predicted: A string predicted for a price tag of a given predicted restaurant, e.g. $$
        :param name: A string name of a given restaurant
        :param cuisine: A string cuisine offered by a given restaurant, e.g. thai, pizza, other, etc.
        :param location: A string city in which a given restaurant is located
        :param menu_text: A list of strings each of which is a dish offered by a restaurant
        :param tokens: A list of strings representing the tokenized text of the menu_text
        :param features: A dictionary holding features as keys with their corresponding weights as values
        :param menu_text_as_whole: A string of the whole menu text of a given restaurant as a whole
        """

        self.restaurant_id = restaurant_id
        self.gold = gold
        self.predicted = predicted
        self.name = name
        self.cuisine = cuisine
        self.location = location
        if menu_text is None:
            menu_text = []
        self.menu_text = menu_text
        if tokens is None:
            tokens = []
        self.tokens = tokens
        if features is None:
            features = {}
        self.features = features
        self.menu_text_as_whole = menu_text_as_whole

    def __str__(self):
        """
        String method
        :return: A string representation of an instance of this class
        """
        return (
                "Restaurant No "
                + str(self.restaurant_id)
                + ": "
                + " gold tag: "
                + self.gold
                + ", predicted tag: "
                + self.predicted
                + ", name: "
                + self.name
                + ", cuisine: "
                + self.cuisine
                + ", location: "
                + self.location
                + ", menu: "
                + str(self.menu_text)
                + ", tokens: "
                + str(self.tokens)
                + ", features: "
                + str(self.features)
                + ", menu text as a whole: "
                + self.menu_text_as_whole
        )
