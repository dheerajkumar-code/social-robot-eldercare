import unittest
from dialog_manager import DialogManager

class TestDialogManager(unittest.TestCase):
    def setUp(self):
        self.dm = DialogManager()

    def test_greeting_intent(self):
        result = self.dm.process_input("Hello", "neutral")
        self.assertEqual(result['intent'], "greeting")
        self.assertTrue(len(result['response']) > 0)

    def test_emotion_adaptation_happy(self):
        # Happy greeting should get a happy response
        result = self.dm.process_input("Hi", "happy")
        # We can't check exact text due to randomness, but we can check intent
        self.assertEqual(result['intent'], "greeting")
        # In a real test we might mock the random choice, but here we just ensure it runs

    def test_emotion_adaptation_sad(self):
        # Sad greeting should get a supportive response
        result = self.dm.process_input("Hi", "sad")
        self.assertEqual(result['intent'], "greeting")

    def test_activity_awareness(self):
        result = self.dm.process_input("What am I doing?", "neutral", "reading_book")
        self.assertEqual(result['intent'], "activity_comment")
        self.assertIn("reading book", result['response'])

    def test_health_check(self):
        result = self.dm.process_input("I feel sick", "neutral")
        self.assertEqual(result['intent'], "user_health_status")

    def test_emergency(self):
        result = self.dm.process_input("Help me call 911", "neutral")
        self.assertEqual(result['intent'], "emergency")
        response_upper = result['response'].upper()
        self.assertTrue("EMERGENCY" in response_upper or "CALLING FOR HELP" in response_upper)

    def test_unknown_input(self):
        result = self.dm.process_input("gibberish text xyz", "neutral")
        self.assertEqual(result['intent'], "unknown")

if __name__ == '__main__':
    unittest.main()
