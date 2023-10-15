import unittest
from app import predict_sentiment

class TestPredictSentiment(unittest.TestCase):

    def test_positive_sentiment(self):
        result = predict_sentiment("I love this movie!")
        self.assertEqual(result, "Positive")

    def test_negative_sentiment(self):
        result = predict_sentiment("This movie is awful.")
        self.assertEqual(result, "Negative")

if _name_ == '_main_':
    unittest.main()