import unittest
import importlib.util

spec = importlib.util.spec_from_file_location('sc', 'c:/Users/Administrator/Downloads/Assignment/NLP-ASSIGNMENT---CT052-3-M-NLP/CodeForQuestion-1/spell_correction_system.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

class TestSpellCorrection(unittest.TestCase):

    def test_typo_suggestion(self):
        checker = m.AdvancedSpellChecker()
        sample = 'patient was admitted to the hospital. the patient reported chest pain.'
        checker.train(sample)
        suggestions = checker.get_suggestions('pationt', 'the pationt has chest pain')
        self.assertTrue(any(s.corrected == 'patient' for s in suggestions), f"Got suggestions: {suggestions}")

    def test_real_word_confusion(self):
        checker = m.AdvancedSpellChecker()
        sample = 'patient was admitted to the hospital. patient was admitted too the ward. patient was given medication to treat cough.'
        checker.train(sample)
        suggestions = checker.get_suggestions('too', 'patient was admitted too the hospital')
        # Real-word detector should prefer 'to' in context
        self.assertTrue(any(s.corrected == 'to' for s in suggestions), f"Got suggestions: {suggestions}")

if __name__ == '__main__':
    unittest.main()
