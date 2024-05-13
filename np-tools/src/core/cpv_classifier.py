import glob
from skops.io import load
import spacy
import logging
from config.config import Config


class CpvClassifier(object):
    def __init__(self, logger: logging.Logger) -> None:

        # Set logger
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

        self.base_path = Config.BASE_PATH
        self.cpv_codes = Config.CPV_CODES
        self.nlp = spacy.load('es_dep_news_trf')
        self.model, self.vectorizer = self.load_model_from_hf()

    def load_model_from_hf(self):
        try:
            model_files = glob.glob(
                self.base_path + '**/model.skops', recursive=True)
            vectorizer_files = glob.glob(
                self.base_path + '**/vectorizer.skops', recursive=True)

            if not model_files or not vectorizer_files:
                self.logger.error(
                    f"Model or vectorizer files not found in {self.base_path}")
                return None, None

            self.logger.info(f"-- -- Model files found: {model_files}")
            self.logger.info(
                f"-- -- Vectorizer files found: {vectorizer_files}")

            model_path = model_files[0]
            vectorizer_path = vectorizer_files[0]

            model = load(model_path)
            vectorizer = load(vectorizer_path)
            self.logger.info("Model and vectorizer loaded successfully")
            return model, vectorizer

        except Exception as e:
            print(f"Error loading models from local storage: {e}")
            return None, None

    def preprocess_text(self, text):
        doc = self.nlp(text.lower())
        words = [
            token.lemma_ for token in doc if not token.is_punct and not token.is_stop]
        return ' '.join(words)

    def predict_description(self, description):
        processed_description = self.preprocess_text(description)
        description_vectorized = self.vectorizer.transform(
            [processed_description])
        probs = self.model.predict_proba(description_vectorized)

        max_prob = max(probs[0])
        predicted_code_index = list(probs[0]).index(max_prob)
        predicted_code = self.cpv_codes[predicted_code_index]

        return {
            "cpv_predicted": predicted_code,
            "probability": round(max_prob, 2)
        }
