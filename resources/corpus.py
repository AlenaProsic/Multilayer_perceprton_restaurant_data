# package resources

"""
    Class to represent a corpus of restaurants
"""
from resources.restaurant import Restaurant

class Corpus:

    price_tag_set = (
        "$",
        "$$",
        "$$$",
        "$$$$",
    )  # set of price_tags to be predicted

    def __init__(self):
        """
        Constructor
        """
        self.restaurants = []

    def read_gold_file(self, filename):
        """
        Read a file with gold standard values and NO predicted values
        :param filename: A filename
        """
        try:
            if filename:
                with open(filename, "r") as f:
                    # read all tweets and assign ids
                    id = 1
                    for line in f:
                        if line != "\n":
                            fields = line.split("\t")
                            restaurant_gold = Restaurant(
                                id,
                                fields[0].strip(),
                                "NONE",
                                fields[1].strip(),
                                fields[2].strip(),
                                fields[3].strip(),
                                fields[4].strip().split(";"),
                            )  # restaurant_id, gold tag, predicted tag, name, cuisine, location, menu_text
                            tokens_menu_text = []
                            for dish in restaurant_gold.menu_text:
                                tokens_menu_text.append(dish.split(" "))
                            restaurant_gold.tokens = [
                                x
                                for sublist in tokens_menu_text
                                for x in sublist
                            ]  # menu tokens
                            restaurant_gold.menu_text_as_whole = " ".join(
                                restaurant_gold.menu_text
                            )  # menu text as one string
                            self.restaurants.append(restaurant_gold)
                            id += 1
                        else:
                            f.close()
        except IOError as error:
            print(error)

    def read_predicted_file(self, filename):
        """
        Read a file with predicted values and NO annotation values
        If list restaurants exists, it will be updated to contain gold and predicted annotation values
        :param filename: A filename
        """
        try:
            if filename:
                with open(filename, "r") as f:
                    i = 0
                    for line in f:
                        if line != "\n":
                            restaurant_predicted = self.restaurants[i]
                            restaurant_predicted.predicted = line.rstrip()
                            self.restaurants[i] = restaurant_predicted
                            i += 1
                        else:
                            f.close()
        except IOError as error:
            print(error)

# c = Corpus()
# c.read_gold_file("/home/PycharmProjects/Lab_V2/src/data/menu_dev.txt")
# c.read_predicted_file("src/data/predicted_data/menu_dev-predicted.txt")
# for res in c.restaurants:
#      print(res.bigrams)
